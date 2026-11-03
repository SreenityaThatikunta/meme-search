"""Test doubles that stand in for CLIP, FAISS and MongoDB.

The real search path needs a 350MB model, two FAISS indexes and a running
MongoDB. The production code defers those imports into ``load_resources``, which
lets the suite swap in these lightweight fakes and still exercise the ranking,
id-mapping, hydration and HTTP layers.
"""

import pytest

from services import search as search_service


class FakeIndex:
    """Minimal stand-in for ``faiss.IndexFlatIP``.

    ``hits`` is the (vector_id, score) list this index "finds", best first.
    """

    def __init__(self, hits, ntotal=None):
        self._hits = list(hits)
        self.ntotal = ntotal if ntotal is not None else len(self._hits)

    def search(self, embedding, k):
        selected = self._hits[:k]
        ids = [vector_id for vector_id, _ in selected]
        scores = [score for _, score in selected]
        # FAISS pads short result sets with -1 ids; mimic that.
        while len(ids) < k:
            ids.append(-1)
            scores.append(0.0)
        return [scores], [ids]


class FakeCollection:
    """In-memory stand-in for a MongoDB collection."""

    def __init__(self, docs):
        self._docs = {doc["meme_id"]: doc for doc in docs}
        self.find_calls = 0

    def find(self, query, projection=None):
        self.find_calls += 1
        wanted = query["meme_id"]["$in"]
        return [dict(self._docs[mid]) for mid in wanted if mid in self._docs]


def make_doc(meme_id, filename, ocr_text=""):
    return {
        "meme_id": meme_id,
        "image_path": f"data/images/images/{filename}",
        "ocr_text": ocr_text,
        "vector_id": int(meme_id.split("_")[-1]),
    }


@pytest.fixture
def loaded_search(monkeypatch):
    """Install fake resources into the search module and hand them back."""
    text_index = FakeIndex([(0, 0.90), (1, 0.70), (2, 0.50)])
    image_index = FakeIndex([(1, 0.80), (3, 0.60), (0, 0.40)])
    id_map = {
        "0": "meme_000000",
        "1": "meme_000001",
        "2": "meme_000002",
        "3": "meme_000003",
    }
    collection = FakeCollection(
        [
            make_doc("meme_000000", "image_1.jpg", "one does not simply"),
            make_doc("meme_000001", "image_2.jpg", "distracted boyfriend"),
            make_doc("meme_000002", "image_3.jpg", ""),
            make_doc("meme_000003", "image_4.png", "this is fine"),
        ]
    )

    monkeypatch.setattr(search_service, "_model", object())
    monkeypatch.setattr(search_service, "_text_index", text_index)
    monkeypatch.setattr(search_service, "_image_index", image_index)
    monkeypatch.setattr(search_service, "_id_map", id_map)
    monkeypatch.setattr(search_service, "_collection", collection)
    # Queries never reach CLIP in tests; the fake indexes ignore the embedding.
    monkeypatch.setattr(search_service, "encode_query", lambda query: [[0.0] * 512])

    return {
        "text_index": text_index,
        "image_index": image_index,
        "id_map": id_map,
        "collection": collection,
    }
