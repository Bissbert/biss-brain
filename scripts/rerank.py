"""Cross-encoder reranking. Public API: rerank(query, candidates, top_k) -> list[dict].

Primary model: BAAI/bge-reranker-v2-m3 via FlagEmbedding.
Fallback (if FlagEmbedding unavailable): cross-encoder/ms-marco-MiniLM-L-6-v2 via
sentence-transformers (already installed as a transitive dep of FlagEmbedding).

The reranker is loaded lazily on first call and cached as a module-level singleton.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("biss-brain.rerank")

# Module-level singleton — set on first call to _get_reranker()
_reranker: Any = None
_reranker_kind: str = ""  # "flag" | "sbert" | "none"


def _get_reranker() -> tuple[Any, str]:
    """Lazy-load the reranker. Returns (model, kind)."""
    global _reranker, _reranker_kind
    if _reranker is not None or _reranker_kind == "none":
        return _reranker, _reranker_kind

    # Try FlagEmbedding primary
    try:
        from FlagEmbedding import FlagReranker  # type: ignore[import]

        log.info("loading BAAI/bge-reranker-v2-m3 via FlagEmbedding...")
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        _reranker_kind = "flag"
        log.info("bge-reranker-v2-m3 loaded")
        return _reranker, _reranker_kind
    except Exception as e:  # noqa: BLE001
        log.warning("FlagEmbedding not available or model load failed (%s); trying sbert fallback", e)

    # Try sentence-transformers fallback
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]

        log.info("loading cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        _reranker_kind = "sbert"
        log.info("ms-marco-MiniLM-L-6-v2 loaded")
        return _reranker, _reranker_kind
    except Exception as e2:  # noqa: BLE001
        log.warning("sentence-transformers fallback also failed (%s); reranking disabled", e2)

    _reranker_kind = "none"
    return None, "none"


def _score_flag(model: Any, query: str, texts: list[str]) -> list[float]:
    """Score using FlagReranker. Returns list of floats (higher = more relevant)."""
    pairs = [[query, t] for t in texts]
    scores = model.compute_score(pairs, normalize=True)
    # compute_score may return a single float when len==1
    if isinstance(scores, float):
        return [scores]
    return list(scores)


def _score_sbert(model: Any, query: str, texts: list[str]) -> list[float]:
    """Score using sentence-transformers CrossEncoder."""
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs)
    try:
        return [float(s) for s in scores]
    except TypeError:
        return [float(scores)]


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank *candidates* by relevance to *query*; return top_k best.

    Each candidate must have at least ``id`` and ``text`` keys.
    If the reranker is unavailable (FlagEmbedding + sbert both absent), the
    original list is returned unchanged (first top_k elements).

    Skips reranking entirely when len(candidates) <= top_k.
    """
    if not candidates:
        return candidates

    if len(candidates) <= top_k:
        return candidates

    model, kind = _get_reranker()

    if kind == "none" or model is None:
        log.debug("reranker unavailable; returning first %d candidates unchanged", top_k)
        return candidates[:top_k]

    texts = [c["text"] for c in candidates]

    try:
        if kind == "flag":
            scores = _score_flag(model, query, texts)
        else:
            scores = _score_sbert(model, query, texts)
    except Exception as e:  # noqa: BLE001
        log.warning("reranker scoring failed (%s); trying sbert fallback", e)
        # If flag failed, attempt sbert as a one-time fallback
        if kind == "flag":
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import]
                sbert_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                scores = _score_sbert(sbert_model, query, texts)
            except Exception as e2:  # noqa: BLE001
                log.warning("sbert fallback also failed (%s); returning original order", e2)
                return candidates[:top_k]
        else:
            return candidates[:top_k]

    for s, c in zip(scores, candidates):
        c["_rerank_score"] = float(s)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]
