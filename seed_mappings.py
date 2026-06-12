"""Bootstrap exam_mappings.csv with synthetic variants of each catalog
description.  Goal: cross the ~100-example threshold that turns the
LightGBM reranker on, using shorthand/abbreviation swaps that mirror how
real exam orders are phrased in the wild.

Each variant is written as:
    <variant query>,<code>,### seeded by seed_mappings.py
"""
from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path
from typing import Iterable, List, Set, Tuple

SRC = Path("exam_codes.csv")
DST = Path("exam_mappings.csv")

# ---- text-level substitutions ----------------------------------------------
# Each rule: (pattern, list of replacements).  Replacements are applied
# independently — we cartesian-product across rules to enumerate variants.
# All matching is case-insensitive; output preserves the replacement string
# verbatim.
SUBSTITUTIONS: list[tuple[re.Pattern, list[str]]] = [
    # Contrast phrasing
    (re.compile(r"\bw\s+and\s+wo\s+IV\s+Contrast\b", re.I),
     ["WWO", "WWOC", "with and without contrast", "w/wo contrast", "w wo"]),
    (re.compile(r"\bwo\s+IV\s+Contrast\b", re.I),
     ["WO", "WOC", "without contrast", "no contrast", "wo cont"]),
    (re.compile(r"\bw\s+IV\s+Contrast\b", re.I),
     ["W", "WC", "with contrast", "w cont", "w contrast"]),

    # Spine shorthand
    (re.compile(r"\bCervical\s+Spine\b", re.I),
     ["C-Spine", "C Spine", "Cspine", "Cervical"]),
    (re.compile(r"\bThoracic\s+Spine\b", re.I),
     ["T-Spine", "T Spine", "Tspine", "Thoracic"]),
    (re.compile(r"\bLumbar\s+Spine\b", re.I),
     ["L-Spine", "L Spine", "Lspine", "Lumbar", "LS Spine"]),

    # Modality aliases
    (re.compile(r"^CT\b", re.I), ["CAT Scan", "CT Scan"]),
    (re.compile(r"^MR\b(?!\s*Angio)", re.I),  # MR but not "MR Angio" (MRA)
     ["MRI"]),
    (re.compile(r"^MR\s+(\w+)\s+Angio\b", re.I), ["MRA \\1", "MR Angiogram \\1"]),
    (re.compile(r"^CT\s+(\w+)\s+Angio\b", re.I), ["CTA \\1", "CT Angiogram \\1"]),
    (re.compile(r"^XR\b", re.I),
     ["X-Ray", "Xray", "Radiograph"]),
    (re.compile(r"^US\b", re.I),
     ["Ultrasound", "Sono", "Sonogram"]),
    (re.compile(r"^NM\b", re.I),
     ["Nuc Med", "Nuclear", "Nuclear Medicine", "Nuc Med Scan"]),
    (re.compile(r"^FL\b", re.I),
     ["Fluoro", "Fluoroscopy"]),
    (re.compile(r"^IR\b", re.I),
     ["Interventional", "IR Procedure"]),
    (re.compile(r"^XA\b", re.I),
     ["Angio", "Angiogram", "Angiography"]),
    (re.compile(r"^MG\b", re.I),
     ["Mammo", "Mammogram", "Mammography"]),
    (re.compile(r"^DXA\b", re.I),
     ["Bone Density", "DEXA", "Bone Densitometry"]),
    (re.compile(r"^PT/CT\b", re.I),
     ["PET CT", "PET/CT", "PET-CT"]),

    # Common anatomy abbreviations
    (re.compile(r"\bAbdomen\b", re.I), ["ABD", "Belly"]),
    (re.compile(r"\bAbdomen/Pelvis\b", re.I), ["ABD/PELVIS", "Abd Pelvis", "AP"]),
    (re.compile(r"\bExtremity\b", re.I), ["Ext"]),

    # Anatomy synonyms — clinical/colloquial alternates that don't share tokens
    # with the canonical body-part name.  These break the synthetic-seed bias
    # where every training query has high token overlap with the catalog text.
    (re.compile(r"\bHead\b", re.I), ["Brain", "Cranial", "Cerebral"]),
    (re.compile(r"\bHeart\b", re.I), ["Cardiac"]),
    (re.compile(r"\bChest\b", re.I), ["Thorax"]),
    (re.compile(r"\bLung\b", re.I), ["Pulmonary"]),
    (re.compile(r"\bKidney\b", re.I), ["Renal"]),
    (re.compile(r"\bLiver\b", re.I), ["Hepatic"]),
    (re.compile(r"\bSpleen\b", re.I), ["Splenic"]),
    (re.compile(r"\bSpine\b", re.I), ["Back"]),
    (re.compile(r"\bPelvis\b", re.I), ["Pelvic"]),
    (re.compile(r"\bShoulder\b", re.I), ["AC Joint"]),
    (re.compile(r"\bFoot\b", re.I), ["Pedal"]),
    (re.compile(r"\bForearm\b", re.I), ["Radius Ulna"]),
    (re.compile(r"\bNeck\b", re.I), ["Cervical Soft Tissue"]),
    (re.compile(r"\bBreast\b", re.I), ["Mammary"]),
    (re.compile(r"\bProstate\b", re.I), ["Prostatic"]),
    (re.compile(r"\bBladder\b", re.I), ["Vesical"]),
    (re.compile(r"\bUterus\b", re.I), ["Uterine"]),

    # Laterality
    (re.compile(r"\bBilateral\b", re.I), ["BILAT", "B/L", "Both", "BI-LATERAL"]),
    (re.compile(r"\bLeft\b", re.I), ["LT", "L"]),
    (re.compile(r"\bRight\b", re.I), ["RT", "R"]),

    # View counts
    (re.compile(r"\b(\d+)\+?\s+Views?\b", re.I), ["\\1 V", "\\1V", "\\1 view"]),
]


def apply_one(pat: re.Pattern, replacement: str, text: str) -> str:
    """Apply a single substitution, honoring backrefs."""
    return pat.sub(replacement, text)


def variants_for(desc: str, max_per_code: int = 8) -> Set[str]:
    """Generate variants of a description by chaining 1 or 2 substitutions.

    We cap chain depth at 2 to keep the variant count manageable; each chain
    is one (text, replacement) -> (text, replacement) sequence using rules
    whose patterns currently match the text.
    """
    out: Set[str] = set()

    # ----- depth-1: single substitution -----
    for pat, repls in SUBSTITUTIONS:
        if not pat.search(desc):
            continue
        for r in repls:
            v = apply_one(pat, r, desc).strip()
            if v and v.lower() != desc.lower():
                out.add(_collapse_ws(v))

    # ----- depth-2: pairs of substitutions (modality swap + contrast swap, etc.) -----
    matching = [(p, repls) for p, repls in SUBSTITUTIONS if p.search(desc)]
    for (p1, r1s), (p2, r2s) in itertools.combinations(matching, 2):
        for r1, r2 in itertools.product(r1s, r2s):
            step1 = apply_one(p1, r1, desc)
            step2 = apply_one(p2, r2, step1).strip()
            if step2 and step2.lower() != desc.lower():
                out.add(_collapse_ws(step2))

    # Cap so a heavily-substitutable description doesn't blow up.  Sort first
    # so the cap is deterministic.
    if len(out) > max_per_code:
        return set(sorted(out)[:max_per_code])
    return out


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build():
    if not SRC.exists():
        raise SystemExit(f"missing source catalog: {SRC}")

    rows_out: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str]] = set()  # (normalized variant, code)

    with SRC.open(encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            code = (row.get("Code") or "").strip()
            desc = (row.get("description") or "").strip()
            if not code or not desc:
                continue

            # always include the verbatim description as a self-mapping
            entries: List[str] = [desc]
            entries.extend(variants_for(desc))

            for v in entries:
                key = (v.lower(), code)
                if key in seen:
                    continue
                seen.add(key)
                rows_out.append((v, code, "### seeded by seed_mappings.py"))

    with DST.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for r in rows_out:
            wr.writerow(r)

    print(f"Wrote {len(rows_out)} seed mappings to {DST}")
    by_code = {}
    for v, c, _ in rows_out:
        by_code[c] = by_code.get(c, 0) + 1
    print(f"Covers {len(by_code)} unique codes")
    counts = sorted(by_code.values())
    print(f"Mappings per code — min: {counts[0]}, median: {counts[len(counts)//2]}, max: {counts[-1]}")


if __name__ == "__main__":
    build()
