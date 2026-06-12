"""
In-memory store for LLM suggestions pending human review.
"""

import threading
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional


_pending: List[Dict[str, Any]] = []
_lock = threading.Lock()


def _qkey(query: str) -> str:
    """Normalized key for de-duplicating by query (case- and spacing-insensitive)."""
    return " ".join((query or "").split()).upper()


def add_pending(
    query: str,
    suggested_code: str,
    suggested_description: Optional[str],
    match_score: int,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Add a suggestion to the review queue, keyed by query.

    If the same query is already pending, the existing item is returned
    unchanged -- the panel only ever shows one entry per query, so a repeated
    failing search never has to be reviewed (or accepted) more than once.
    """
    query = (query or "").strip()
    key = _qkey(query)
    with _lock:
        for existing in _pending:
            if _qkey(existing.get("query", "")) == key:
                return existing
        item = {
            "id": uuid.uuid4().hex[:12],
            "query": query,
            "suggested_code": (suggested_code or "").strip(),
            "suggested_description": (suggested_description or "").strip(),
            "match_score": match_score,
            "confidence": confidence,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        _pending.append(item)
        return item


def get_pending() -> List[Dict[str, Any]]:
    """Return all pending items."""
    with _lock:
        return list(_pending)


def agree(ids: List[str]) -> List[tuple]:
    """
    Resolve agreed items to their (query, code) pairs and drop them from the
    pending queue. Any other pending entry sharing the same query is dropped too
    -- once a query is mapped, its duplicates are resolved as well.

    Persisting the pairs is the caller's job (done as a single batched write so
    accepting many suggestions is fast). Returns the list of (query, code) pairs.
    """
    to_add: List[tuple] = []
    with _lock:
        for item_id in ids:
            for item in _pending:
                if item.get("id") == item_id:
                    to_add.append((item["query"], item["suggested_code"]))
                    break
        agreed_keys = {_qkey(q) for q, _ in to_add}
        _pending[:] = [it for it in _pending
                       if _qkey(it.get("query", "")) not in agreed_keys]

    return to_add


def clear_all() -> int:
    """Remove all pending items. Returns count removed."""
    with _lock:
        n = len(_pending)
        _pending.clear()
        return n


def disagree(item_id: str, correct_code: str, upsert_mapping_fn) -> bool:
    """
    Disagree: use correct_code instead, add to mappings, remove from pending.
    Any other pending entry for the same query is removed too.
    Returns True if item was found and processed.
    """
    query = None
    with _lock:
        for item in _pending:
            if item.get("id") == item_id:
                query = item["query"]
                break
        if query is not None:
            key = _qkey(query)
            _pending[:] = [it for it in _pending
                           if _qkey(it.get("query", "")) != key]

    if query is None:
        return False

    upsert_mapping_fn(query, correct_code.strip(), "### corrected by AI review (disagree)")
    return True
