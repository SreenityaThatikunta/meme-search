"""Retrieval metric maths."""

import pytest

from services.metrics import hit_at_k, reciprocal_rank, summarise


def test_reciprocal_rank_of_first_position_is_one():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0


def test_reciprocal_rank_counts_from_one_not_zero():
    assert reciprocal_rank(["a", "b", "c"], "b") == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b", "c"], "c") == pytest.approx(1 / 3)


def test_reciprocal_rank_of_absent_gold_is_zero():
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_reciprocal_rank_of_empty_ranking_is_zero():
    assert reciprocal_rank([], "a") == 0.0


def test_hit_at_k_respects_the_cutoff():
    ranking = ["a", "b", "c"]
    assert hit_at_k(ranking, "c", 3) is True
    # "c" sits at rank 3, so a cutoff of 2 must exclude it.
    assert hit_at_k(ranking, "c", 2) is False


@pytest.mark.parametrize("bad_k", [0, -1])
def test_hit_at_k_rejects_non_positive_cutoffs(bad_k):
    with pytest.raises(ValueError, match="k must be"):
        hit_at_k(["a"], "a", bad_k)


def test_summarise_computes_recall_and_mrr():
    rankings = [["a", "x", "y"], ["x", "b", "y"], ["x", "y", "z"]]
    golds = ["a", "b", "c"]
    result = summarise(rankings, golds, ks=(1, 3))

    assert result["queries"] == 3
    # Only the first query puts gold at rank 1.
    assert result["recall@1"] == pytest.approx(1 / 3)
    # The first two queries find gold within three results; the third never does.
    assert result["recall@3"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_summarise_is_monotonic_in_k():
    rankings = [["x", "a"], ["b", "y"]]
    result = summarise(rankings, ["a", "b"], ks=(1, 2, 5))
    assert result["recall@1"] <= result["recall@2"] <= result["recall@5"]


def test_summarise_of_empty_query_set():
    assert summarise([], [], ks=(1, 5)) == {"queries": 0}


def test_summarise_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        summarise([["a"], ["b"]], ["a"])
