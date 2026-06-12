"""
LightGBM pairwise (LambdaRank) reranker.

Replaces the per-class RandomForest in matcher.py. Instead of asking
"which of 2,064 classes is this?" -- a question we have ~2 examples per class
to answer -- we ask "given this query and these N candidate codes, rank them."

That framing turns each training mapping into N candidate pairs (the gold
code + N-1 negatives), so 4,000 mappings produce ~200k pairwise rows. Far
more signal per example, and unseen codes are no longer invisible: they're
scored from features, not class identity.

Features (per query-candidate pair):
    [0]  rule_score          -- the existing _score_code output (rich domain prior)
    [1]  text_sim_char       -- TF-IDF char 2-4 gram cosine
    [2]  text_sim_word       -- TF-IDF word 1-2 gram cosine
    [3]  embed_sim           -- sentence-transformer cosine
    [4]  modality_match      -- 1/0
    [5]  view_match          -- 1/0/-1
    [6]  contrast_match      -- 1/0/-1
    [7]  laterality_match    -- 1/0/0.5/-1
    [8]  body_overlap        -- ratio
    [9]  token_overlap_jacc  -- jaccard
    [10] fuzzy_ratio
    [11] norm_edit_distance
    [12] proc_keyword_score
    [13] generic_proc_score
    [14] alias_overlap       -- ratio of query tokens in code's training-alias vocab
    [15] code_n_train        -- log1p of training examples for this code (popularity prior)
    [16] is_exact_prefix     -- 1 if code description startswith query
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_NAMES = [
    "rule_score",
    "text_sim_char",
    "text_sim_word",
    "embed_sim",
    "modality_match",
    "view_match",
    "contrast_match",
    "laterality_match",
    "body_overlap",
    "token_overlap_jacc",
    "fuzzy_ratio",
    "norm_edit_distance",
    "proc_keyword_score",
    "generic_proc_score",
    "alias_overlap",
    "code_n_train",
    "is_exact_prefix",
]


@dataclass
class RerankerArtifact:
    model: object                          # lgb.Booster
    code_train_counts: Dict[str, int]      # code -> #training examples (for the prior feature)
    feature_names: List[str]
    trained_on: int                        # number of query groups used
    # Optional calibrator: maps the top1 raw score (or margin) -> P(top1 correct).
    # Lets the app replace the magic min_score threshold with an actual
    # probability the runtime can reason about ("fall back to LLM if confidence
    # < 0.7"). Stored as a sklearn IsotonicRegression-compatible object.
    calibrator: object = None
    calibrator_kind: str = "margin"  # "margin" | "top1_raw"

    def save(self, path: Path | str):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path | str) -> "RerankerArtifact":
        import joblib
        art = joblib.load(path)
        # Older artifacts predate calibration fields; default them in.
        if not hasattr(art, "calibrator"):
            art.calibrator = None
        if not hasattr(art, "calibrator_kind"):
            art.calibrator_kind = "margin"
        return art

    def confidence(self, top1_raw: float, top2_raw: Optional[float] = None) -> Optional[float]:
        """Return calibrated P(top1 correct), or None if no calibrator."""
        if self.calibrator is None:
            return None
        if self.calibrator_kind == "margin":
            margin = top1_raw - (top2_raw if top2_raw is not None else 0.0)
            return float(self.calibrator.predict([margin])[0])
        return float(self.calibrator.predict([top1_raw])[0])


def fit_calibrator(
    matcher,
    embedding_index,
    artifact: RerankerArtifact,
    eval_examples: Sequence[Tuple[str, str]],
    kind: str = "margin",
):
    """Fit isotonic regression mapping reranker score signal -> P(top1 correct).

    Uses eval_examples (held-out, never seen by the reranker) so calibration is
    honest. Result is attached to the artifact in-place.
    """
    from sklearn.isotonic import IsotonicRegression

    raw_xs: List[float] = []
    raw_ys: List[int] = []
    for norm_query, gold in eval_examples:
        if gold not in matcher.code_by_id:
            continue
        query_info = matcher._parse_query(norm_query, norm_query)
        query_info = _normalize_query_info_for_feature(query_info)
        cands = candidate_pool(matcher, embedding_index, norm_query, query_info)
        if not cands:
            continue
        feats = build_pair_features(
            matcher, embedding_index, norm_query, cands,
            query_info=query_info,
            code_train_counts=artifact.code_train_counts,
        )
        scores = artifact.model.predict(feats)
        order = np.argsort(-scores)
        top1_idx = order[0]
        top1_code = cands[top1_idx]
        top1_raw = float(scores[top1_idx])
        top2_raw = float(scores[order[1]]) if len(order) > 1 else 0.0
        x = (top1_raw - top2_raw) if kind == "margin" else top1_raw
        raw_xs.append(x)
        raw_ys.append(1 if top1_code == gold else 0)

    if len(raw_xs) < 30:
        print(f"[calibrator] not enough eval examples ({len(raw_xs)}); skipping")
        return

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_xs, raw_ys)
    artifact.calibrator = iso
    artifact.calibrator_kind = kind
    print(f"[calibrator] fit on {len(raw_xs)} eval examples (kind={kind})")


def _normalize_query_info_for_feature(qi: Dict) -> Dict:
    """Backfill flags the reranker reads but _parse_query may not set."""
    qi.setdefault("body_parts", set())
    qi.setdefault("procedure_keywords", set())
    qi.setdefault("knowledge_tokens", [])
    qi.setdefault("generic_procedure", None)
    qi.setdefault("has_plurality", False)
    return qi


def build_pair_features(
    matcher,
    embedding_index,
    query: str,
    candidate_code_ids: Sequence[str],
    query_info: Optional[Dict] = None,
    query_vec=None,
    query_word_vec=None,
    text_sims: Optional[np.ndarray] = None,
    word_sims: Optional[np.ndarray] = None,
    embed_query_vec: Optional[np.ndarray] = None,
    code_train_counts: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Build an (n_candidates, n_features) feature matrix.

    Reuses matcher's existing extractors and TF-IDF index so we don't duplicate
    domain logic. The reranker just learns weights over these signals.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from matcher import (
        fuzzy_similarity, normalized_edit_distance,
        procedure_keyword_score, generic_procedure_score,
        detect_generic_procedure, view_count_score,
    )

    normalized = matcher._normalize_text(query) if not query.isupper() or " " not in query else matcher._normalize_text(query)
    if query_info is None:
        query_info = matcher._parse_query(normalized, query)
    query_info = _normalize_query_info_for_feature(query_info)

    code_by_id = matcher.code_by_id
    codes_in_order = [code_by_id[c] for c in candidate_code_ids if c in code_by_id]
    valid_mask = [c in code_by_id for c in candidate_code_ids]

    code_id_to_idx_in_codes = {c.code: i for i, c in enumerate(matcher.codes)}
    cand_idxs_in_codes = [code_id_to_idx_in_codes[c.code] for c in codes_in_order]

    if text_sims is None:
        if query_vec is None:
            query_vec = matcher.vectorizer.transform([normalized])
        all_text = cosine_similarity(query_vec, matcher.tfidf_matrix)[0]
    else:
        all_text = text_sims
    if word_sims is None and matcher.vectorizer_word is not None:
        if query_word_vec is None:
            query_word_vec = matcher.vectorizer_word.transform([normalized])
        all_word = cosine_similarity(query_word_vec, matcher.tfidf_matrix_word)[0]
    elif word_sims is not None:
        all_word = word_sims
    else:
        all_word = np.zeros(len(matcher.codes))

    # Embedding sims for this candidate set only
    if embedding_index is not None:
        if embed_query_vec is None:
            embed_query_vec = embedding_index.encode_query(normalized)
        embed_sims_array = embedding_index.similarity(
            embed_query_vec, [c.code for c in codes_in_order]
        )
    else:
        embed_sims_array = np.zeros(len(codes_in_order), dtype=np.float32)

    query_tokens = set(normalized.split())
    proc_keywords = query_info.get("procedure_keywords", set())
    generic_proc = query_info.get("generic_procedure")
    qb_parts = query_info.get("body_parts") or set()

    rows = []
    for j, code in enumerate(codes_in_order):
        code_row = cand_idxs_in_codes[j]
        text_sim_char = float(all_text[code_row])
        text_sim_word = float(all_word[code_row])
        embed_sim = float(embed_sims_array[j])

        rule_score = matcher._score_code(
            query_info, code, code_row, normalized,
            return_log=False, text_sim=text_sim_char,
        )

        modality_match = 1.0 if (query_info.get("modality") and query_info["modality"] == code.modality) else 0.0

        if query_info.get("views") is None:
            view_match = 0.0
        elif query_info["views"] == code.view_count:
            view_match = 1.0
        elif getattr(code, "view_is_minimum", False) and query_info["views"] > code.view_count:
            view_match = 1.0
        else:
            view_match = -1.0

        qc = query_info.get("contrast", "UNKNOWN")
        if qc == "UNKNOWN":
            contrast_match = 0.0
        elif qc == code.contrast:
            contrast_match = 1.0
        else:
            contrast_match = -1.0

        ql = query_info.get("laterality", "NONE")
        if ql == "NONE":
            lat_match = 0.0
        elif ql == code.laterality:
            lat_match = 1.0
        elif code.laterality == "NONE":
            lat_match = 0.5
        else:
            lat_match = -1.0

        if qb_parts and code.body_regions:
            overlap = len(qb_parts & set(code.body_regions))
            body_overlap = overlap / len(qb_parts)
        else:
            body_overlap = 0.0

        if query_tokens and code.tokens:
            tok_overlap = len(query_tokens & code.tokens) / len(query_tokens | code.tokens)
        else:
            tok_overlap = 0.0

        fuzzy_r = fuzzy_similarity(normalized, code.normalized_desc)
        norm_edit = normalized_edit_distance(normalized, code.normalized_desc)
        proc_score = procedure_keyword_score(proc_keywords, code.normalized_desc)
        gp_score = float(generic_procedure_score(generic_proc, code.normalized_desc)) / 150.0 if generic_proc else 0.0

        # alias_overlap: how many query tokens appear in this code's known training-time aliases
        alias_tokens = matcher.code_alias_tokens[code_row] if code_row < len(matcher.code_alias_tokens) else set()
        if alias_tokens:
            alias_overlap = len(query_tokens & alias_tokens) / max(1, len(alias_tokens))
        else:
            alias_overlap = 0.0

        n_train = (code_train_counts or {}).get(code.code, 0)
        code_n_train = math.log1p(n_train)

        is_prefix = 1.0 if code.normalized_desc.startswith(normalized) else 0.0

        rows.append([
            float(rule_score),
            text_sim_char,
            text_sim_word,
            embed_sim,
            modality_match,
            view_match,
            contrast_match,
            lat_match,
            body_overlap,
            tok_overlap,
            float(fuzzy_r),
            float(norm_edit),
            float(proc_score),
            gp_score,
            float(alias_overlap),
            code_n_train,
            is_prefix,
        ])

    # If any candidates were missing from code_by_id, pad zeros for those slots
    if not all(valid_mask):
        full = np.zeros((len(candidate_code_ids), len(FEATURE_NAMES)), dtype=np.float32)
        out_idx = 0
        for i, ok in enumerate(valid_mask):
            if ok:
                full[i] = rows[out_idx]
                out_idx += 1
        return full
    return np.array(rows, dtype=np.float32)


def candidate_pool(
    matcher,
    embedding_index,
    normalized_query: str,
    query_info: Dict,
    embed_k: int = 50,
    tfidf_k: int = 50,
    text_sims: Optional[np.ndarray] = None,
    word_sims: Optional[np.ndarray] = None,
    embed_top_k: Optional[Sequence[Tuple[str, float]]] = None,
) -> List[str]:
    """Build the candidate set for a query.

    Strategy: union of
      1. All codes matching the parsed modality (if any) -- recall floor
      2. Top-K from embedding cosine
      3. Top-K from char-TFIDF cosine
      4. Top-K from word-TFIDF cosine

    The text_sims / word_sims / embed_top_k args let callers pass in
    precomputed values, so the batched training loop can compute them
    once across all queries instead of per-query.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    out: List[str] = []
    seen = set()

    def add(cid):
        if cid in seen:
            return
        seen.add(cid)
        out.append(cid)

    query_mod = query_info.get("modality")
    if query_mod:
        for code in matcher.codes:
            if code.modality == query_mod:
                add(code.code)

    if embedding_index is not None:
        if embed_top_k is None:
            embed_top_k = embedding_index.top_k(normalized_query, k=embed_k)
        for cid, _ in embed_top_k:
            add(cid)

    if text_sims is None:
        query_vec = matcher.vectorizer.transform([normalized_query])
        text_sims = cosine_similarity(query_vec, matcher.tfidf_matrix)[0]
    top_text = np.argpartition(-text_sims, min(tfidf_k, len(text_sims) - 1))[:tfidf_k]
    for idx in top_text:
        add(matcher.codes[idx].code)

    if matcher.vectorizer_word is not None and matcher.tfidf_matrix_word is not None:
        if word_sims is None:
            query_word_vec = matcher.vectorizer_word.transform([normalized_query])
            word_sims = cosine_similarity(query_word_vec, matcher.tfidf_matrix_word)[0]
        top_word = np.argpartition(-word_sims, min(tfidf_k, len(word_sims) - 1))[:tfidf_k]
        for idx in top_word:
            add(matcher.codes[idx].code)

    return out


def _mine_one_query(args):
    """Worker function: mine hard negatives for one query."""
    (matcher, embedding_index, norm_query, gold_code, code_train_counts,
     embed_q_row, text_sims_row, word_sims_row, embed_topk_row, top_k_arg,
     seed_model) = args

    if gold_code not in matcher.code_by_id:
        return None
    query_info = matcher._parse_query(norm_query, norm_query)
    query_info = _normalize_query_info_for_feature(query_info)
    cands = candidate_pool(
        matcher, embedding_index, norm_query, query_info,
        embed_k=30, tfidf_k=30,
        text_sims=text_sims_row, word_sims=word_sims_row,
        embed_top_k=embed_topk_row,
    )
    if not cands:
        return None
    feats = build_pair_features(
        matcher, embedding_index, norm_query, cands,
        query_info=query_info,
        text_sims=text_sims_row, word_sims=word_sims_row,
        embed_query_vec=embed_q_row,
        code_train_counts=code_train_counts,
    )
    scores = seed_model.predict(feats)
    order = np.argsort(-scores)
    wrong_top = [cands[i] for i in order[:top_k_arg] if cands[i] != gold_code]
    return (norm_query, wrong_top[:top_k_arg]) if wrong_top else None


def mine_self_hard_negatives(
    matcher,
    embedding_index,
    seed_artifact: "RerankerArtifact",
    top_k: int = 5,
    n_jobs: int = -1,
) -> Dict[str, List[str]]:
    """For each training example, score candidates with the seed reranker and
    collect the top-K *wrong* predictions as hard negatives.

    Two-pass training (seed model -> mine its mistakes -> retrain) is a
    cheap and very effective form of hard-negative mining when no explicit
    rejection signal exists.

    Parallelized + batch-precomputed like train_reranker.
    """
    from joblib import Parallel, delayed

    train_examples = list(getattr(matcher, "training_examples", []))
    if not train_examples:
        return {}

    queries = [q for q, _ in train_examples]
    embed_q, text_sims, word_sims, embed_topk = _precompute_query_signals(
        matcher, embedding_index, queries, embed_k=30,
    )

    work = []
    for i, (norm_query, gold_code) in enumerate(train_examples):
        work.append((
            matcher, embedding_index, norm_query, gold_code,
            seed_artifact.code_train_counts,
            embed_q[i] if embed_q is not None else None,
            text_sims[i],
            word_sims[i] if word_sims is not None else None,
            embed_topk[i] if embed_topk is not None else None,
            top_k, seed_artifact.model,
        ))

    t = time.time()
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=64, verbose=0)(
        delayed(_mine_one_query)(w) for w in work
    )
    print(f"[mining] parallel pass in {time.time()-t:.1f}s (n_jobs={n_jobs})")

    out: Dict[str, List[str]] = {}
    for res in results:
        if res is not None:
            q, negs = res
            out[q] = negs
    return out


def _precompute_query_signals(matcher, embedding_index, queries: Sequence[str],
                              embed_k: int = 40):
    """Encode all queries in batches and pre-compute the heavy similarity
    matrices once, instead of inside the per-query loop.

    Returns:
      embed_q   : (N, dim) query embeddings (or None if no embedding index)
      text_sims : (N, n_codes) char-TFIDF cosine sims
      word_sims : (N, n_codes) word-TFIDF cosine sims (or None)
      embed_topk: list-of-list of (cid, sim) tuples per query (or None)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    t = time.time()
    embed_q = None
    embed_topk = None
    if embedding_index is not None and embedding_index.matrix is not None:
        embed_q = embedding_index.encode_queries(queries)
        # Batched cosine via single matmul.  Both sides are L2-normalized.
        sims_all = embed_q @ embedding_index.matrix.T  # (N, n_codes)
        k = min(embed_k, sims_all.shape[1])
        # argpartition for top-k per row, then sort within the top-k
        topk_idx = np.argpartition(-sims_all, k - 1, axis=1)[:, :k]
        embed_topk = []
        for i in range(sims_all.shape[0]):
            row = topk_idx[i]
            order = row[np.argsort(-sims_all[i, row])]
            embed_topk.append([(embedding_index.code_ids[j], float(sims_all[i, j])) for j in order])
    print(f"[reranker] batch-encoded {len(queries)} queries in {time.time()-t:.1f}s")

    t = time.time()
    qv = matcher.vectorizer.transform(list(queries))
    text_sims = cosine_similarity(qv, matcher.tfidf_matrix).astype(np.float32)
    word_sims = None
    if matcher.vectorizer_word is not None and matcher.tfidf_matrix_word is not None:
        qwv = matcher.vectorizer_word.transform(list(queries))
        word_sims = cosine_similarity(qwv, matcher.tfidf_matrix_word).astype(np.float32)
    print(f"[reranker] batch TF-IDF sims in {time.time()-t:.1f}s "
          f"(shape={text_sims.shape})")

    return embed_q, text_sims, word_sims, embed_topk


def _build_one_query_features(args):
    """Worker function: build features for a single training query.

    Designed to be picklable for joblib.Parallel.  Inputs are precomputed
    so workers don't redo embedding/TF-IDF math.
    """
    (matcher, embedding_index, norm_query, gold_code, extra_negs,
     embed_q_row, text_sims_row, word_sims_row, embed_topk_row,
     candidates_per_query, code_train_counts, rng_seed) = args

    if gold_code not in matcher.code_by_id:
        return None

    query_info = matcher._parse_query(norm_query, norm_query)
    query_info = _normalize_query_info_for_feature(query_info)

    candidates = candidate_pool(
        matcher, embedding_index, norm_query, query_info,
        embed_k=40, tfidf_k=40,
        text_sims=text_sims_row, word_sims=word_sims_row,
        embed_top_k=embed_topk_row,
    )
    if gold_code not in candidates:
        candidates.append(gold_code)
    if extra_negs:
        for hn in extra_negs:
            if hn != gold_code and hn in matcher.code_by_id and hn not in candidates:
                candidates.append(hn)

    rng = np.random.default_rng(rng_seed)
    negs = [c for c in candidates if c != gold_code]
    if len(negs) > candidates_per_query - 1:
        chosen = rng.choice(negs, size=candidates_per_query - 1, replace=False)
        sampled = [gold_code] + list(chosen)
    else:
        sampled = [gold_code] + negs
    rng.shuffle(sampled)
    gold_pos = sampled.index(gold_code)

    feats = build_pair_features(
        matcher, embedding_index, norm_query, sampled,
        query_info=query_info,
        text_sims=text_sims_row, word_sims=word_sims_row,
        embed_query_vec=embed_q_row,
        code_train_counts=code_train_counts,
    )
    return feats, gold_pos, len(sampled)


def train_reranker(
    matcher,
    embedding_index,
    extra_hard_negatives: Optional[Dict[str, List[str]]] = None,
    candidates_per_query: int = 30,
    num_boost_round: int = 400,
    n_jobs: int = -1,
) -> RerankerArtifact:
    """Train LightGBM LambdaRank on matcher.training_examples.

    For each (query, gold_code) pair we sample candidates_per_query negatives
    from the candidate pool. Optionally augment with extra_hard_negatives
    (e.g. mined from match_history rejections).

    Speedups:
      * Batch-encode all query embeddings + TF-IDF in one pass (avoids per-query
        sentence-transformer overhead).
      * Parallelize the per-query feature build across `n_jobs` workers
        (n_jobs=-1 uses all cores).
    """
    import lightgbm as lgb
    from joblib import Parallel, delayed

    train_examples = list(getattr(matcher, "training_examples", []))
    if not train_examples:
        raise RuntimeError("matcher has no training_examples to learn from")

    code_train_counts: Dict[str, int] = {}
    for _, c in train_examples:
        code_train_counts[c] = code_train_counts.get(c, 0) + 1

    print(f"[reranker] building features for {len(train_examples)} training queries...")
    t0 = time.time()

    # --- batch precompute everything that doesn't depend on candidates ---
    queries = [q for q, _ in train_examples]
    embed_q, text_sims, word_sims, embed_topk = _precompute_query_signals(
        matcher, embedding_index, queries, embed_k=40,
    )

    # --- assemble per-query work units ---
    extra_hard_negatives = extra_hard_negatives or {}
    work = []
    for i, (norm_query, gold_code) in enumerate(train_examples):
        work.append((
            matcher, embedding_index, norm_query, gold_code,
            extra_hard_negatives.get(norm_query),
            embed_q[i] if embed_q is not None else None,
            text_sims[i],
            word_sims[i] if word_sims is not None else None,
            embed_topk[i] if embed_topk is not None else None,
            candidates_per_query, code_train_counts,
            42 + i,  # seeded RNG per query for determinism
        ))

    # --- parallel feature build ---
    t = time.time()
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=64, verbose=0)(
        delayed(_build_one_query_features)(w) for w in work
    )
    print(f"[reranker] parallel feature build in {time.time()-t:.1f}s "
          f"(n_jobs={n_jobs})")

    feat_rows: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[int] = []
    skipped = 0
    for res in results:
        if res is None:
            skipped += 1
            continue
        feats, gold_pos, n_sampled = res
        feat_rows.append(feats)
        lbl = np.zeros(n_sampled, dtype=np.int32)
        lbl[gold_pos] = 1
        labels.extend(lbl.tolist())
        groups.append(n_sampled)

    if not feat_rows:
        raise RuntimeError("no training rows built (all candidates invalid?)")

    X = np.vstack(feat_rows)
    y = np.array(labels, dtype=np.int32)
    g = np.array(groups, dtype=np.int32)
    print(f"[reranker] feature build done: {X.shape}, {len(g)} groups, "
          f"{skipped} skipped, {time.time() - t0:.1f}s")

    dtrain = lgb.Dataset(X, label=y, group=g, feature_name=FEATURE_NAMES)
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 5, 10],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
    )
    print("[reranker] training LightGBM LambdaRank...")
    fit_t0 = time.time()
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    print(f"[reranker] fit done in {time.time() - fit_t0:.1f}s")

    return RerankerArtifact(
        model=model,
        code_train_counts=code_train_counts,
        feature_names=FEATURE_NAMES,
        trained_on=len(g),
    )


def rerank_predict(
    artifact: RerankerArtifact,
    matcher,
    embedding_index,
    normalized_query: str,
    candidate_code_ids: Sequence[str],
    query_info: Optional[Dict] = None,
) -> Tuple[List[Tuple[str, float, np.ndarray]], np.ndarray]:
    """Score candidates for a single query. Returns (ranked, raw_feature_matrix)."""
    feats = build_pair_features(
        matcher, embedding_index, normalized_query, list(candidate_code_ids),
        query_info=query_info,
        code_train_counts=artifact.code_train_counts,
    )
    scores = artifact.model.predict(feats)
    order = np.argsort(-scores)
    ranked = [(candidate_code_ids[i], float(scores[i]), feats[i]) for i in order]
    return ranked, feats
