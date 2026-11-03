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

    Both indexes are inner-product indexes over L2-normalised CLIP embeddings,
    so their scores are cosine similarities on the same ``[-1, 1]`` scale and can
    be blended directly. A meme found by only one index keeps its own score
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


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Drop repeated meme_ids, keeping the highest score for each."""
    best: dict[str, float] = {}
    for meme_id, score in candidates:
        if meme_id not in best or score > best[meme_id]:
            best[meme_id] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))
