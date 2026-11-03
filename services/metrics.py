"""Retrieval metrics.

Dependency-free so the maths can be unit tested without a model, an index or a
database — the same reason :mod:`services.fusion` is kept separate.
"""

import math
from collections.abc import Mapping, Sequence


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


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of an ordered gain list."""
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, 1))


def ndcg_at_k(ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """Normalised DCG@k over graded relevance judgments.

    Gains are exponential (``2**rel - 1``), the TREC convention, so a highly
    relevant result is worth meaningfully more than a partially relevant one
    rather than merely twice as much.

    Documents absent from ``judgments`` score 0. With pooled judging that is the
    standard assumption: a document no system retrieved is treated as
    irrelevant.

    Returns 0.0 when no relevant document exists for the query, so such queries
    neither reward nor punish a system.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    gains = [2 ** judgments.get(meme_id, 0) - 1 for meme_id in ranked_ids[:k]]
    ideal = sorted((2**rel - 1 for rel in judgments.values()), reverse=True)[:k]

    best = dcg(ideal)
    if best == 0:
        return 0.0
    return dcg(gains) / best


def precision_at_k(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, int],
    k: int,
    threshold: int = 2,
) -> float:
    """Fraction of the first ``k`` results judged at least ``threshold``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(judgments.get(m, 0) >= threshold for m in top) / len(top)


def cohen_kappa(a: Sequence[int], b: Sequence[int], weights: str = "none") -> float:
    """Inter-annotator agreement between two graded judgment sets.

    Raw agreement overstates reliability when one grade dominates: two
    annotators who both label 80% of items "relevant" agree 80% of the time by
    chance alone. Kappa subtracts that expected agreement.

    Interpretation is conventional rather than exact: below 0.2 is poor, 0.4-0.6
    moderate, 0.6-0.8 substantial, above 0.8 near-perfect.

    Args:
        a: One annotator's grades.
        b: The other's, aligned item for item.
        weights: ``"none"`` for standard kappa, which treats every disagreement
            as equally wrong, or ``"quadratic"`` which penalises a 0-vs-2
            disagreement far more than 1-vs-2. Grades here are ordinal, so
            quadratic is usually the fairer reading.

    Returns:
        Kappa in ``[-1, 1]``. 1.0 is perfect agreement, 0.0 is chance-level, and
        negative values mean worse than chance. Returns 1.0 when both annotators
        used a single identical grade throughout, where kappa is otherwise
        undefined.
    """
    if len(a) != len(b):
        raise ValueError(f"judgment sets must align: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cannot compute agreement over zero items")
    if weights not in ("none", "quadratic"):
        raise ValueError(f"weights must be 'none' or 'quadratic', got {weights!r}")

    categories = sorted(set(a) | set(b))
    total = len(a)
    index = {label: position for position, label in enumerate(categories)}
    size = len(categories)

    observed = [[0.0] * size for _ in range(size)]
    for first, second in zip(a, b, strict=True):
        observed[index[first]][index[second]] += 1 / total

    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(size)) for c in range(size)]

    def penalty(r: int, c: int) -> float:
        if r == c:
            return 0.0
        if weights == "none" or size == 1:
            return 1.0
        return ((categories[r] - categories[c]) ** 2) / ((categories[-1] - categories[0]) ** 2)

    disagreement = sum(penalty(r, c) * observed[r][c] for r in range(size) for c in range(size))
    expected = sum(
        penalty(r, c) * row_totals[r] * col_totals[c] for r in range(size) for c in range(size)
    )

    if expected == 0:
        # Both annotators used one identical grade throughout.
        return 1.0 if disagreement == 0 else 0.0
    return 1.0 - disagreement / expected


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
