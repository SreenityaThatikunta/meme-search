"""Ranking maths for hybrid retrieval."""

import pytest

from services.fusion import dedupe_candidates, fuse_scores


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
