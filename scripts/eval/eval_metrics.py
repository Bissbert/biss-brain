"""Retrieval metrics, answer grading and bootstrap statistics for the eval harness."""
from __future__ import annotations

import math
import random
import re
from typing import Iterable, Sequence


# ── grading ──────────────────────────────────────────────────────────────
#
# A question's `answers` is a list of term groups. A retrieved text is
# relevant when, for at least one group, every term occurs in it
# (case-insensitive; a term wrapped in /.../ is a regex). Ids in `gold_ids`
# also count as relevant (secondary key; ids do not survive re-distillation).

def _term_hit(term: str, text: str) -> bool:
    if len(term) > 2 and term.startswith("/") and term.endswith("/"):
        return re.search(term[1:-1], text, re.IGNORECASE) is not None
    return term.lower() in text.lower()


def is_relevant(text: str, chunk_id: str, question: dict) -> bool:
    if chunk_id and any(chunk_id.startswith(g) for g in question.get("gold_ids", [])):
        return True
    return any(group and all(_term_hit(t, text) for t in group) for group in question.get("answers", []))


def relevance_vector(results: Sequence[dict], question: dict) -> list[int]:
    return [int(is_relevant(r.get("text", ""), r.get("id", ""), question)) for r in results]


def is_answerable(question: dict) -> bool:
    return bool(question.get("answers") or question.get("gold_ids"))


# ── per-query metrics ────────────────────────────────────────────────────

def hit_at_k(rel: Sequence[int], k: int) -> float:
    """1 if any relevant result is in the top k (recall@k with one-fact questions)."""
    return float(any(rel[:k]))


def reciprocal_rank(rel: Sequence[int]) -> float:
    for i, r in enumerate(rel, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: Sequence[int], k: int) -> float:
    """Binary-gain nDCG. Ideal = all relevant results found, ranked first."""
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel[:k], 1))
    n_rel = sum(rel)
    if n_rel == 0:
        return 0.0
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(n_rel, k) + 1))
    return dcg / idcg


def precision(rel: Sequence[int]) -> float | None:
    return sum(rel) / len(rel) if rel else None


# ── aggregation ──────────────────────────────────────────────────────────

def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(xs)
    stats = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return stats[int(alpha / 2 * n)], stats[min(n - 1, int((1 - alpha / 2) * n))]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> dict:
    """Per-question paired comparison of variant b against a.

    Returns the mean delta (b - a), its 95% CI and the share of resamples in
    which b does not beat a (a one-sided p-value proxy).
    """
    assert len(a) == len(b), "paired bootstrap needs the same questions"
    d = [y - x for x, y in zip(a, b)]
    if not d:
        return {"delta": float("nan"), "ci": (float("nan"), float("nan")), "p_not_better": float("nan")}
    rng = random.Random(seed)
    m = len(d)
    stats = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {
        "delta": sum(d) / m,
        "ci": (stats[int(0.025 * n)], stats[min(n - 1, int(0.975 * n))]),
        "p_not_better": sum(1 for s in stats if s <= 0) / n,
    }
