"""HTTP contract of the FastAPI service."""

import pytest
from fastapi.testclient import TestClient

import main
from services import search as search_service


@pytest.fixture
def client(monkeypatch, loaded_search):
    # Startup must not try to load a real model or reach MongoDB.
    monkeypatch.setattr(search_service, "load_resources", lambda **kwargs: None)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_reports_loaded_resources(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["text_index_vectors"] == 3
    assert body["available_modes"] == ["text", "image", "hybrid"]


def test_search_returns_ranked_results(client):
    response = client.post("/search/text", json={"query": "exams", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert set(body[0]) == {"meme_id", "image_path", "image_url", "ocr_text", "score"}


def test_search_defaults_to_hybrid_mode(client):
    body = client.post("/search/text", json={"query": "exams", "top_k": 1}).json()
    assert body[0]["meme_id"] == "meme_000001"


def test_mode_can_be_overridden(client):
    body = client.post("/search/text", json={"query": "exams", "top_k": 1, "mode": "text"}).json()
    assert body[0]["meme_id"] == "meme_000000"


def test_empty_query_is_rejected_by_validation(client):
    assert client.post("/search/text", json={"query": ""}).status_code == 422


def test_missing_query_is_rejected(client):
    assert client.post("/search/text", json={"top_k": 3}).status_code == 422


@pytest.mark.parametrize("top_k", [0, 999])
def test_out_of_range_top_k_is_rejected(client, top_k):
    response = client.post("/search/text", json={"query": "exams", "top_k": top_k})
    assert response.status_code == 422


def test_unknown_mode_is_rejected(client):
    response = client.post("/search/text", json={"query": "exams", "mode": "telepathy"})
    assert response.status_code == 422


def test_out_of_range_text_weight_is_rejected(client):
    response = client.post("/search/text", json={"query": "exams", "text_weight": 1.5})
    assert response.status_code == 422


def test_unavailable_mode_returns_400(client, monkeypatch):
    monkeypatch.setattr(search_service, "_image_index", None)
    response = client.post("/search/text", json={"query": "exams", "mode": "image"})
    assert response.status_code == 400
    assert "image index" in response.json()["detail"]


def test_missing_index_file_returns_503(client, monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("Text index not found at data/text.index")

    monkeypatch.setattr(search_service, "search", raise_missing)
    response = client.post("/search/text", json={"query": "exams"})
    assert response.status_code == 503


def test_unexpected_errors_return_500_without_leaking_internals(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("faiss segfaulted on shard 3")

    monkeypatch.setattr(search_service, "search", boom)
    response = client.post("/search/text", json={"query": "exams"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal search error"


def test_openapi_schema_is_served(client):
    assert client.get("/openapi.json").status_code == 200
