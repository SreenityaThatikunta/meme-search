"""Retrieval metrics.

Dependency-free so the maths can be unit tested without a model, an index or a
database — the same reason :mod:`services.fusion` is kept separate.
"""

from collections.abc import Sequence


def reciprocal_rank(ranked_ids: Sequence[str], gold_id: str) -> float:
    """Return ``1 / rank`` of ``gold_id``, or 0.0 when it is absent.

    With exactly one relevant document per query, the mean of this over a query
    set is MRR.
    """
    for position, candidate in enumerate(ranked_ids, start=1):
        if candidate == gold_id:
            return 1.0 / position
    return 0.0


def hit_at_k(ranked_ids: Sequence[str], gold_id: str, k: int) -> bool:
    """Whether ``gold_id`` appears in the first ``k`` results."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return gold_id in ranked_ids[:k]


def summarise(
    rankings: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Aggregate Recall@k and MRR over a query set.

    With a single relevant document per query, Recall@k equals Hit-rate@k and
    equals Precision@k times k, so only one of them is worth reporting.

    Args:
        rankings: One ranked list of meme ids per query.
        gold_ids: The relevant meme id for each query, same order.
        ks: Cutoffs to report.

    Returns:
        ``{"recall@1": ..., "recall@5": ..., "mrr": ..., "queries": n}``.
    """
    if len(rankings) != len(gold_ids):
        raise ValueError(f"rankings and gold_ids must align: {len(rankings)} vs {len(gold_ids)}")
    if not rankings:
        return {"queries": 0}

    total = len(rankings)
    results: dict[str, float] = {"queries": float(total)}
    for k in ks:
        hits = sum(hit_at_k(r, g, k) for r, g in zip(rankings, gold_ids, strict=True))
        results[f"recall@{k}"] = hits / total
    results["mrr"] = (
        sum(reciprocal_rank(r, g) for r, g in zip(rankings, gold_ids, strict=True)) / total
    )
    return results
