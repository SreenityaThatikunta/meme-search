"""Retrieval behaviour: modes, id mapping, hydration and input validation."""

import pytest

import config
from services import search as search_service


def test_text_mode_ranks_by_the_text_index(loaded_search):
    results = search_service.search("exams", top_k=3, mode="text")
    assert [r["meme_id"] for r in results] == [
        "meme_000000",
        "meme_000001",
        "meme_000002",
    ]
    assert results[0]["score"] == pytest.approx(0.90)


def test_image_mode_ranks_by_the_image_index(loaded_search):
    results = search_service.search("exams", top_k=2, mode="image")
    assert [r["meme_id"] for r in results] == ["meme_000001", "meme_000003"]


def test_hybrid_mode_blends_both_indexes(loaded_search):
    results = search_service.search("exams", top_k=4, mode="hybrid", text_weight=0.5)
    # meme_000001 scores 0.70 on text and 0.80 on image, so it wins on fusion
    # despite meme_000000 leading the text index outright.
    assert results[0]["meme_id"] == "meme_000001"
    assert results[0]["score"] == pytest.approx(0.75)


def test_hybrid_weight_shifts_the_winner(loaded_search):
    text_heavy = search_service.search("exams", top_k=1, mode="hybrid", text_weight=1.0)
    image_heavy = search_service.search("exams", top_k=1, mode="hybrid", text_weight=0.0)
    assert text_heavy[0]["meme_id"] == "meme_000000"
    assert image_heavy[0]["meme_id"] == "meme_000001"


def test_results_carry_image_urls_built_from_the_public_base(loaded_search):
    results = search_service.search("exams", top_k=1, mode="text")
    assert results[0]["image_url"] == f"{config.PUBLIC_BASE_URL}/images/image_1.jpg"


def test_metadata_is_fetched_in_one_round_trip(loaded_search):
    search_service.search("exams", top_k=3, mode="text")
    assert loaded_search["collection"].find_calls == 1


def test_top_k_is_respected(loaded_search):
    assert len(search_service.search("exams", top_k=2, mode="text")) == 2


def test_top_k_is_capped_at_the_configured_maximum(loaded_search, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOP_K", 2)
    assert len(search_service.search("exams", top_k=50, mode="text")) == 2


def test_vectors_missing_from_the_id_map_are_skipped(loaded_search, monkeypatch):
    monkeypatch.setattr(search_service, "_id_map", {"0": "meme_000000"})
    results = search_service.search("exams", top_k=3, mode="text")
    assert [r["meme_id"] for r in results] == ["meme_000000"]


def test_memes_missing_from_mongo_are_skipped(loaded_search, monkeypatch):
    from tests.conftest import FakeCollection, make_doc

    # Simulates FAISS and Mongo built from different dataset snapshots.
    monkeypatch.setattr(
        search_service,
        "_collection",
        FakeCollection([make_doc("meme_000001", "image_2.jpg")]),
    )
    results = search_service.search("exams", top_k=3, mode="text")
    assert [r["meme_id"] for r in results] == ["meme_000001"]


def test_faiss_padding_ids_are_ignored(loaded_search, monkeypatch):
    from tests.conftest import FakeIndex

    # ntotal lies high so the index pads the result set with -1.
    monkeypatch.setattr(search_service, "_text_index", FakeIndex([(0, 0.9)], ntotal=5))
    results = search_service.search("exams", top_k=5, mode="text")
    assert [r["meme_id"] for r in results] == ["meme_000000"]


def test_blank_query_returns_nothing(loaded_search):
    assert search_service.search("   ", top_k=5, mode="text") == []


def test_unknown_mode_is_rejected(loaded_search):
    with pytest.raises(ValueError, match="mode must be one of"):
        search_service.search("exams", mode="semantic")


def test_non_positive_top_k_is_rejected(loaded_search):
    with pytest.raises(ValueError, match="top_k must be"):
        search_service.search("exams", top_k=0)


def test_image_mode_without_an_image_index_fails_clearly(loaded_search, monkeypatch):
    monkeypatch.setattr(search_service, "_image_index", None)
    with pytest.raises(ValueError, match="needs the image index"):
        search_service.search("exams", mode="hybrid")


def test_available_modes_reflects_the_loaded_indexes(loaded_search, monkeypatch):
    assert search_service.available_modes() == ("text", "image", "hybrid")
    monkeypatch.setattr(search_service, "_image_index", None)
    assert search_service.available_modes() == ("text",)


def test_legacy_search_text_helper_still_works(loaded_search):
    results = search_service.search_text("exams", top_k=1)
    assert results[0]["meme_id"] == "meme_000000"
