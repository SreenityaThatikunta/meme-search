"""Ranking maths for hybrid retrieval."""

import pytest

from services.fusion import dedupe_candidates, fuse_ranks, fuse_scores


def test_equal_weights_average_the_two_signals():
    fused = dict(fuse_scores([("a", 0.8)], [("a", 0.4)], text_weight=0.5))
    assert fused["a"] == pytest.approx(0.6)


def test_agreement_outranks_a_single_strong_signal():
    # "b" is weaker in each index individually but is found by both.
    fused = fuse_scores(
        [("a", 0.9), ("b", 0.6)],
        [("b", 0.6), ("c", 0.8)],
        text_weight=0.5,
    )
    assert [meme_id for meme_id, _ in fused][0] == "b"


def test_text_weight_one_ignores_the_image_index():
    fused = dict(fuse_scores([("a", 0.5)], [("b", 0.99)], text_weight=1.0))
    assert fused == {"a": pytest.approx(0.5), "b": pytest.approx(0.0)}


def test_text_weight_zero_ignores_the_text_index():
    fused = dict(fuse_scores([("a", 0.99)], [("b", 0.5)], text_weight=0.0))
    assert fused["a"] == pytest.approx(0.0)
    assert fused["b"] == pytest.approx(0.5)


def test_results_are_sorted_by_descending_score():
    fused = fuse_scores([("a", 0.1), ("b", 0.9)], [], text_weight=1.0)
    assert [meme_id for meme_id, _ in fused] == ["b", "a"]


def test_ties_break_deterministically_on_meme_id():
    fused = fuse_scores([("z", 0.5), ("a", 0.5)], [], text_weight=1.0)
    assert [meme_id for meme_id, _ in fused] == ["a", "z"]


@pytest.mark.parametrize("bad_weight", [-0.1, 1.5])
def test_weight_outside_unit_range_is_rejected(bad_weight):
    with pytest.raises(ValueError, match="text_weight"):
        fuse_scores([], [], text_weight=bad_weight)


def test_empty_inputs_produce_no_candidates():
    assert fuse_scores([], [], text_weight=0.5) == []


def test_dedupe_keeps_the_highest_score_per_meme():
    assert dedupe_candidates([("a", 0.3), ("a", 0.7), ("b", 0.5)]) == [
        ("a", 0.7),
        ("b", 0.5),
    ]


# --- Reciprocal rank fusion -------------------------------------------------
# RRF exists because text-to-text and text-to-image cosines differ ~3x in
# magnitude on this dataset, which a weighted sum cannot reconcile.


def test_rrf_ignores_score_magnitudes():
    # Identical rank orders, wildly different scales: RRF must agree.
    big = [("a", 0.91), ("b", 0.88)]
    small = [("a", 0.31), ("b", 0.28)]
    assert [m for m, _ in fuse_ranks(big, [], 1.0)] == [m for m, _ in fuse_ranks(small, [], 1.0)]


def test_rrf_survives_the_modality_gap_where_the_sum_fails():
    # Text cosines ~3x the image cosines, as measured on the real indexes. "c"
    # is the image index's top hit but carries a magnitude no text hit can lose to.
    text = [("a", 0.91), ("b", 0.90), ("d", 0.88)]
    image = [("c", 0.31)]

    # Under a weighted sum, every text hit outscores the image index's rank 1,
    # so "c" sinks to last despite topping its own list.
    summed = [m for m, _ in fuse_scores(text, image, text_weight=0.5)]
    assert summed == ["a", "b", "d", "c"]

    # RRF compares positions instead, so rank 1 of each list ranks together.
    ranked = [m for m, _ in fuse_ranks(text, image, text_weight=0.5)]
    assert ranked[:2] == ["a", "c"]


def test_rrf_agreement_beats_a_single_first_place():
    text = [("a", 0.9), ("b", 0.8)]
    image = [("c", 0.9), ("b", 0.8)]
    assert [m for m, _ in fuse_ranks(text, image, text_weight=0.5)][0] == "b"


def test_rrf_weight_one_uses_only_the_text_ranking():
    # Rank order within a list comes from that list's own scores, so "b" leads.
    fused = fuse_ranks([("a", 0.1), ("b", 0.9)], [("z", 0.99)], text_weight=1.0)
    assert [m for m, _ in fused][:2] == ["b", "a"]
    # The image-only candidate contributes nothing at weight 1.0.
    assert dict(fused)["z"] == 0.0


def test_rrf_weight_zero_uses_only_the_image_ranking():
    fused = fuse_ranks([("z", 0.99)], [("a", 0.1), ("b", 0.9)], text_weight=0.0)
    assert [m for m, _ in fused][:2] == ["b", "a"]
    assert dict(fused)["z"] == 0.0


def test_rrf_k_damps_the_top_rank_advantage():
    text, image = [("a", 0.9)], [("b", 0.9)]
    # Larger k flattens the gap between rank 1 of each list.
    tight = dict(fuse_ranks(text, image, 0.5, k=1))
    flat = dict(fuse_ranks(text, image, 0.5, k=1000))
    assert abs(tight["a"] - tight["b"]) <= 1e-9
    assert flat["a"] < tight["a"]


def test_rrf_deduplicates_repeated_candidates():
    # A meme listed twice must be ranked once, by its best position.
    fused = fuse_ranks([("a", 0.5), ("a", 0.9), ("b", 0.4)], [], text_weight=1.0)
    assert [m for m, _ in fused] == ["a", "b"]


@pytest.mark.parametrize("bad_weight", [-0.1, 1.5])
def test_rrf_rejects_weights_outside_the_unit_range(bad_weight):
    with pytest.raises(ValueError, match="text_weight"):
        fuse_ranks([], [], text_weight=bad_weight)


@pytest.mark.parametrize("bad_k", [0, -5])
def test_rrf_rejects_non_positive_k(bad_k):
    with pytest.raises(ValueError, match="k must be positive"):
        fuse_ranks([], [], text_weight=0.5, k=bad_k)
