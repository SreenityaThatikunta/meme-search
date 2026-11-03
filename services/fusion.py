"""Score fusion for hybrid retrieval.

Kept free of torch/faiss/mongo so the ranking logic can be unit tested on its
own, without a model or a database.
"""

from collections.abc import Iterable

# A candidate is a (meme_id, score) pair straight out of a FAISS search.
Candidate = tuple[str, float]


def fuse_scores(
    text_hits: Iterable[Candidate],
    image_hits: Iterable[Candidate],
    text_weight: float = 0.5,
) -> list[Candidate]:
    """Combine OCR-text and image candidates into one ranked list.

    .. warning::
       Although both indexes return cosine similarities in ``[-1, 1]``, the two
       are not comparable across modalities: text-to-text cosines are roughly
       3x larger than text-to-image ones (CLIP's modality gap), so this weighted
       sum is dominated by the text index at almost any weight. Use
       :func:`fuse_ranks` for cross-modal fusion. This function is kept for
       same-modality blending and for the comparison in the evaluation harness. A meme found by only one index keeps its own score
    scaled by that index's weight, which naturally favours memes both views
    agree on.

    Args:
        text_hits: Candidates from the OCR-text index.
        image_hits: Candidates from the image index.
        text_weight: Weight for the text index, in ``[0, 1]``. The image index
            gets ``1 - text_weight``.

    Returns:
        Candidates sorted by fused score, highest first. Ties break on meme_id
        so the ordering is deterministic.
    """
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError(f"text_weight must be in [0, 1], got {text_weight}")

    image_weight = 1.0 - text_weight
    fused: dict[str, float] = {}

    for meme_id, score in text_hits:
        fused[meme_id] = fused.get(meme_id, 0.0) + text_weight * score

    for meme_id, score in image_hits:
        fused[meme_id] = fused.get(meme_id, 0.0) + image_weight * score

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def fuse_ranks(
    text_hits: Iterable[Candidate],
    image_hits: Iterable[Candidate],
    text_weight: float = 0.5,
    k: int = 60,
) -> list[Candidate]:
    """Combine candidates by RANK rather than by score (weighted RRF).

    Prefer this over :func:`fuse_scores` for cross-modal fusion. Although both
    indexes return cosine similarities in ``[-1, 1]``, those similarities are not
    comparable across modalities: measured on this dataset, text-to-text cosines
    average 0.904 while text-to-image cosines average 0.307 — CLIP's *modality
    gap*. A weighted sum of the two is therefore dominated by the text index at
    almost any weight, which is a scaling artefact and not a ranking decision.

    Reciprocal Rank Fusion sidesteps the problem by discarding magnitudes and
    keeping only positions::

        score(d) = w / (k + rank_text(d)) + (1 - w) / (k + rank_image(d))

    Args:
        text_hits: Candidates from the OCR-text index, best first.
        image_hits: Candidates from the image index, best first.
        text_weight: Weight for the text list, in ``[0, 1]``.
        k: RRF damping constant. Larger values flatten the contribution of top
            ranks; 60 is the value from the original RRF paper.

    Returns:
        Candidates sorted by fused RRF score, highest first. Ties break on
        meme_id for determinism. Note the returned scores are RRF scores, not
        cosine similarities, so they are comparable only within one result set.
    """
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError(f"text_weight must be in [0, 1], got {text_weight}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    image_weight = 1.0 - text_weight
    fused: dict[str, float] = {}

    for rank, (meme_id, _) in enumerate(dedupe_candidates(text_hits), start=1):
        fused[meme_id] = fused.get(meme_id, 0.0) + text_weight / (k + rank)

    for rank, (meme_id, _) in enumerate(dedupe_candidates(image_hits), start=1):
        fused[meme_id] = fused.get(meme_id, 0.0) + image_weight / (k + rank)

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Drop repeated meme_ids, keeping the highest score for each."""
    best: dict[str, float] = {}
    for meme_id, score in candidates:
        if meme_id not in best or score > best[meme_id]:
            best[meme_id] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))
