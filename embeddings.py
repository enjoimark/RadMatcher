"""
Embedding-based candidate retriever.

Encodes every catalog code description (plus alias text mined from training
mappings) once with a small sentence-transformer, then ranks candidates by
cosine similarity at query time.

Used as the *first stage* of matching: cheaply narrow ~2,064 codes down to
~50 candidates that the reranker can score with the expensive features.

Two things matter for the self-improvement loop:
  * `add_aliases` lets us extend a code's alias text when a new mapping is
    accepted, without rebuilding the whole encoder.
  * The vector store is persisted to disk so app restarts skip re-encoding.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _pick_device() -> str:
    """Prefer Apple Silicon GPU (MPS) > CUDA > CPU.  ~2-3x faster on M-series."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class EmbeddingIndex:
    """In-memory normalized embedding matrix + parallel code-id list."""

    model_name: str = DEFAULT_MODEL
    code_ids: List[str] = field(default_factory=list)
    code_texts: List[str] = field(default_factory=list)
    matrix: Optional[np.ndarray] = None  # shape (n_codes, dim), L2-normalized
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    _model: object = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- model handling -------------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = _pick_device()
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vecs = model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    # ---- build / update -------------------------------------------------
    def build(self, code_id_to_text: Dict[str, str], aliases: Optional[Dict[str, List[str]]] = None):
        """Encode every code's description + alias-joined text."""
        self.code_ids = list(code_id_to_text.keys())
        self.aliases = {cid: list(aliases.get(cid, [])) if aliases else [] for cid in self.code_ids}
        self.code_texts = [self._compose_text(cid, code_id_to_text[cid]) for cid in self.code_ids]
        with self._lock:
            self.matrix = self._encode(self.code_texts)

    def add_aliases(self, code_id_to_aliases: Dict[str, Iterable[str]], code_id_to_text: Dict[str, str]):
        """Re-encode just the codes whose alias text changed.

        Called from upsert_mapping when a new verified mapping arrives, so the
        retriever picks up the new query phrasing without a full rebuild.
        """
        if self.matrix is None:
            raise RuntimeError("index not built yet")
        cid_to_row = {cid: i for i, cid in enumerate(self.code_ids)}
        changed_rows: List[int] = []
        changed_texts: List[str] = []
        for cid, new_aliases in code_id_to_aliases.items():
            if cid not in cid_to_row:
                # new code -- skip, builder owns full rebuild for unseen codes
                continue
            existing = set(self.aliases.get(cid, []))
            additions = [a for a in new_aliases if a and a not in existing]
            if not additions:
                continue
            self.aliases.setdefault(cid, []).extend(additions)
            row = cid_to_row[cid]
            self.code_texts[row] = self._compose_text(cid, code_id_to_text[cid])
            changed_rows.append(row)
            changed_texts.append(self.code_texts[row])
        if not changed_rows:
            return 0
        new_vecs = self._encode(changed_texts)
        with self._lock:
            self.matrix[changed_rows] = new_vecs
        return len(changed_rows)

    def _compose_text(self, code_id: str, description: str) -> str:
        parts = [description.strip()]
        seen = {description.strip().lower()}
        for alias in self.aliases.get(code_id, []):
            key = alias.strip().lower()
            if key and key not in seen:
                seen.add(key)
                parts.append(alias.strip())
        return " | ".join(parts) if parts else description

    # ---- query ---------------------------------------------------------
    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query])[0]

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Batched query encoding -- one model call for the whole list.

        Use this in training loops where you need vectors for thousands of
        queries; calling encode_query in a loop is much slower because each
        call has model-invocation overhead.
        """
        return self._encode(queries)

    def top_k(self, query: str, k: int = 50, allowed_code_ids: Optional[set] = None,
              query_vec: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        if self.matrix is None:
            return []
        q = query_vec if query_vec is not None else self.encode_query(query)
        with self._lock:
            sims = self.matrix @ q  # both L2-normalized, so dot = cosine
        if allowed_code_ids is not None:
            mask = np.array([cid in allowed_code_ids for cid in self.code_ids])
            sims = np.where(mask, sims, -np.inf)
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            # argpartition is O(n) vs O(n log n) -- worth it given we call this on every match
            part = np.argpartition(-sims, k)[:k]
            order = part[np.argsort(-sims[part])]
        return [(self.code_ids[i], float(sims[i])) for i in order if np.isfinite(sims[i])]

    def similarity(self, query_vec: np.ndarray, code_ids: Sequence[str]) -> np.ndarray:
        """Cosine sim between a precomputed query vector and a subset of codes."""
        if self.matrix is None:
            return np.zeros(len(code_ids), dtype=np.float32)
        cid_to_row = {cid: i for i, cid in enumerate(self.code_ids)}
        rows = [cid_to_row.get(cid, -1) for cid in code_ids]
        out = np.zeros(len(code_ids), dtype=np.float32)
        valid = [(i, r) for i, r in enumerate(rows) if r >= 0]
        if not valid:
            return out
        idxs = np.array([r for _, r in valid])
        with self._lock:
            sub = self.matrix[idxs]
        sims = sub @ query_vec
        for (i, _), s in zip(valid, sims):
            out[i] = float(s)
        return out

    # ---- persistence ---------------------------------------------------
    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            code_ids=np.array(self.code_ids, dtype=object),
            code_texts=np.array(self.code_texts, dtype=object),
            matrix=self.matrix,
            aliases_codes=np.array(list(self.aliases.keys()), dtype=object),
            aliases_values=np.array([
                "\x1f".join(self.aliases[c]) for c in self.aliases.keys()
            ], dtype=object),
            model_name=np.array(self.model_name, dtype=object),
        )

    @classmethod
    def load(cls, path: Path | str) -> "EmbeddingIndex":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        idx = cls(model_name=str(data["model_name"]))
        idx.code_ids = list(data["code_ids"])
        idx.code_texts = list(data["code_texts"])
        idx.matrix = data["matrix"]
        codes = list(data["aliases_codes"])
        values = list(data["aliases_values"])
        idx.aliases = {
            c: ([v for v in s.split("\x1f") if v] if s else [])
            for c, s in zip(codes, values)
        }
        return idx


def build_index_from_matcher(
    matcher,
    cache_path: Optional[Path | str] = None,
    force: bool = False,
) -> EmbeddingIndex:
    """Build (or load from cache) an EmbeddingIndex covering matcher.codes.

    Uses training_examples to seed aliases per code, so the encoder sees the
    real-world phrasings users have validated -- not just the catalog text.
    """
    if cache_path is not None and Path(cache_path).exists() and not force:
        try:
            idx = EmbeddingIndex.load(cache_path)
            existing_codes = set(idx.code_ids)
            current_codes = {c.code for c in matcher.codes}
            if existing_codes == current_codes:
                return idx
            print(f"[embeddings] cache mismatch ({len(existing_codes)} vs {len(current_codes)} codes) — rebuilding")
        except Exception as exc:
            print(f"[embeddings] cache load failed: {exc!r} — rebuilding")

    aliases: Dict[str, List[str]] = {}
    for query, code in getattr(matcher, "training_examples", []):
        aliases.setdefault(code, []).append(query)

    code_id_to_text = {c.code: c.description for c in matcher.codes}
    idx = EmbeddingIndex()
    print(f"[embeddings] encoding {len(code_id_to_text)} codes with {idx.model_name}...")
    t0 = time.time()
    idx.build(code_id_to_text, aliases=aliases)
    print(f"[embeddings] encoded in {time.time() - t0:.1f}s")

    if cache_path is not None:
        idx.save(cache_path)
    return idx
