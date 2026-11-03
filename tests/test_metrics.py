"""Retrieval metric maths."""

import pytest

from services.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    summarise,
)


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


# --- Graded relevance: nDCG and precision ----------------------------------


def test_ndcg_is_one_for_the_ideal_ranking():
    judgments = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], judgments, 3) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking():
    judgments = {"a": 2, "b": 1}
    ideal = ndcg_at_k(["a", "b"], judgments, 2)
    reversed_ = ndcg_at_k(["b", "a"], judgments, 2)
    assert reversed_ < ideal


def test_ndcg_rewards_ranking_the_graded_best_first():
    judgments = {"high": 2, "low": 1}
    assert ndcg_at_k(["high"], judgments, 1) > ndcg_at_k(["low"], judgments, 1)


def test_ndcg_treats_unjudged_documents_as_irrelevant():
    # The pooling assumption: unretrieved documents score nothing.
    assert ndcg_at_k(["unknown"], {"a": 2}, 1) == pytest.approx(0.0)


def test_ndcg_is_zero_when_no_relevant_document_exists():
    # Such a query must neither reward nor punish a system.
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 2) == pytest.approx(0.0)


def test_ndcg_uses_exponential_gain():
    # With 2**rel - 1, one rel=2 hit (gain 3) must beat one rel=1 hit (gain 1)
    # by a factor of three, not two.
    high = ndcg_at_k(["a"], {"a": 2, "b": 2}, 1)
    low = ndcg_at_k(["b"], {"a": 2, "b": 1}, 1)
    assert high > low


def test_ndcg_respects_the_cutoff():
    judgments = {"a": 2, "b": 2}
    # "b" sits outside k=1, so the ranking cannot be ideal at that cutoff.
    assert ndcg_at_k(["x", "a", "b"], judgments, 1) == pytest.approx(0.0)


def test_precision_counts_only_fully_relevant_by_default():
    judgments = {"a": 2, "b": 1, "c": 0}
    # Default threshold 2, so the partially relevant "b" does not count.
    assert precision_at_k(["a", "b", "c"], judgments, 3) == pytest.approx(1 / 3)


def test_precision_threshold_can_include_partial_matches():
    judgments = {"a": 2, "b": 1, "c": 0}
    assert precision_at_k(["a", "b", "c"], judgments, 3, threshold=1) == pytest.approx(2 / 3)


def test_precision_of_an_empty_ranking_is_zero():
    assert precision_at_k([], {"a": 2}, 5) == 0.0


def test_precision_divides_by_the_results_returned_not_k():
    # Only two results exist, both relevant: precision is 1.0, not 2/5.
    assert precision_at_k(["a", "b"], {"a": 2, "b": 2}, 5) == pytest.approx(1.0)


@pytest.mark.parametrize("bad_k", [0, -3])
def test_graded_metrics_reject_non_positive_cutoffs(bad_k):
    with pytest.raises(ValueError, match="k must be"):
        ndcg_at_k(["a"], {"a": 2}, bad_k)
    with pytest.raises(ValueError, match="k must be"):
        precision_at_k(["a"], {"a": 2}, bad_k)
