"""Tests for the JSONL metadata backend.

The point of :class:`JsonlCollection` is that ``services.search`` cannot tell it
from a MongoDB collection. These tests pin the two halves of that: the document
shape (which must match what ``scripts/pipeline/store_metadata.py`` writes into
Mongo) and the ``find`` semantics the search path relies on.
"""

import json

import pytest

import config
from services import search as search_service
from services.metadata import JsonlCollection, flatten_record

RECORDS = [
    {
        "meme_id": "meme_000000",
        "image_path": "data/images/images/image_1.jpg",
        "width": 735,
        "height": 500,
        "text": {"ocr": "one does not simply"},
        "vector_id": 0,
    },
    {
        "meme_id": "meme_000001",
        "image_path": "data/images/images/image_2.png",
        "width": 500,
        "height": 500,
        "text": {"ocr": ""},
        "vector_id": 1,
    },
]


@pytest.fixture
def jsonl_path(tmp_path):
    path = tmp_path / "memes_final.jsonl"
    with open(path, "w") as handle:
        for record in RECORDS:
            handle.write(json.dumps(record) + "\n")
        handle.write("\n")  # a blank trailing line must not become a document
    return path


def test_flatten_lifts_nested_ocr_to_the_field_search_reads():
    # The JSONL nests the caption under text.ocr; the search path reads
    # ocr_text. A drift here costs every result its caption without raising.
    doc = flatten_record(RECORDS[0])
    assert doc["ocr_text"] == "one does not simply"
    assert doc["meme_id"] == "meme_000000"
    assert doc["vector_id"] == 0
    assert doc["image_path"] == "data/images/images/image_1.jpg"


def test_flatten_tolerates_a_record_with_no_text_block():
    doc = flatten_record({"meme_id": "meme_000009"})
    assert doc["ocr_text"] == ""
    assert doc["image_path"] == ""
    assert doc["vector_id"] is None


def test_find_returns_requested_ids_and_skips_unknown_ones(jsonl_path):
    collection = JsonlCollection(jsonl_path)
    assert len(collection) == 2

    found = list(collection.find({"meme_id": {"$in": ["meme_000001", "nope"]}}, {"_id": 0}))
    assert [doc["meme_id"] for doc in found] == ["meme_000001"]


def test_find_with_no_matches_returns_empty(jsonl_path):
    collection = JsonlCollection(jsonl_path)
    assert list(collection.find({"meme_id": {"$in": []}})) == []


def test_find_hands_back_copies_so_callers_cannot_corrupt_the_store(jsonl_path):
    collection = JsonlCollection(jsonl_path)
    first = next(iter(collection.find({"meme_id": {"$in": ["meme_000000"]}})))
    first["ocr_text"] = "mutated"

    second = next(iter(collection.find({"meme_id": {"$in": ["meme_000000"]}})))
    assert second["ocr_text"] == "one does not simply"


def test_missing_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="embed_memes_fast.py"):
        JsonlCollection(tmp_path / "absent.jsonl")


def test_search_hydrates_results_through_the_jsonl_backend(monkeypatch, jsonl_path):
    """End-to-end: the search path works with no database at all."""
    from tests.conftest import FakeIndex

    monkeypatch.setattr(config, "METADATA_BACKEND", "jsonl")
    monkeypatch.setattr(search_service, "_model", object())
    monkeypatch.setattr(search_service, "_text_index", FakeIndex([(0, 0.9), (1, 0.5)]))
    monkeypatch.setattr(search_service, "_image_index", None)
    monkeypatch.setattr(search_service, "_id_map", {"0": "meme_000000", "1": "meme_000001"})
    monkeypatch.setattr(search_service, "_collection", JsonlCollection(jsonl_path))
    monkeypatch.setattr(search_service, "encode_query", lambda query: [[0.0] * 512])

    results = search_service.search("one does not simply", top_k=2, mode="text")

    assert [r["meme_id"] for r in results] == ["meme_000000", "meme_000001"]
    assert results[0]["ocr_text"] == "one does not simply"


def test_health_reports_the_backend_without_pinging_a_database(monkeypatch, jsonl_path):
    monkeypatch.setattr(config, "METADATA_BACKEND", "jsonl")
    monkeypatch.setattr(search_service, "_collection", JsonlCollection(jsonl_path))
    monkeypatch.setattr(search_service, "_model", object())

    report = search_service.health()

    assert report["metadata_backend"] == "jsonl"
    assert report["metadata_connected"] is True
    assert report["metadata_error"] is None
    # Legacy keys stay populated so existing clients keep working.
    assert report["mongo_connected"] is True


def test_health_flags_an_unloaded_store(monkeypatch):
    monkeypatch.setattr(config, "METADATA_BACKEND", "jsonl")
    monkeypatch.setattr(search_service, "_collection", None)

    report = search_service.health()

    assert report["metadata_connected"] is False
    assert "not loaded" in report["metadata_error"]


def test_unknown_backend_is_rejected_by_name(monkeypatch):
    monkeypatch.setattr(config, "METADATA_BACKEND", "sqlite")
    with pytest.raises(ValueError, match="'mongo' or 'jsonl'"):
        search_service._open_metadata_store()
