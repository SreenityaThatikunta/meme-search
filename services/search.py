"""Query-time search: CLIP text encoding, FAISS retrieval, MongoDB hydration.

Three retrieval modes share one CLIP text embedding of the query:

``text``
    Compare against embeddings of the OCR text scraped off each meme. Good when
    the user remembers the caption ("one does not simply").
``image``
    Compare against embeddings of the meme images themselves. CLIP maps text and
    images into the same space, so this finds memes that *look* like the query
    even when they carry no readable text.
``hybrid``
    Query both indexes and blend the cosine similarities (see
    :mod:`services.fusion`). This is the default because the two views fail in
    different ways: OCR misses stylised text, and image embeddings miss jokes
    that live entirely in the caption.

The heavy imports (torch, clip, faiss, pymongo) happen inside
:func:`load_resources` rather than at module import time, so the API module and
the test suite can be imported without a model or a database present.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

import config
from services.fusion import Candidate, dedupe_candidates, fuse_ranks, fuse_scores

logger = logging.getLogger(__name__)

VALID_MODES = ("text", "image", "hybrid")

# Lazily populated by load_resources().
_model = None
_text_index = None
_image_index = None
_id_map: dict[str, str] | None = None
_collection = None


class ResourcesNotLoaded(RuntimeError):
    """Raised when a search is attempted before resources are available."""


def load_resources(require_image_index: bool = False) -> None:
    """Load the CLIP model, FAISS indexes, id map and MongoDB handle.

    Idempotent: already-loaded resources are left alone, so this is safe to call
    from both application startup and an individual search.

    Args:
        require_image_index: Fail instead of warning when the image index is
            missing. Image and hybrid search are unavailable without it.
    """
    global _model, _text_index, _image_index, _id_map, _collection

    import clip  # noqa: PLC0415 - deliberately deferred, see module docstring
    import faiss  # noqa: PLC0415

    if _model is None:
        logger.info("Loading CLIP model %s on %s", config.CLIP_MODEL, config.DEVICE)
        _model, _ = clip.load(config.CLIP_MODEL, device=config.DEVICE)
        _model.eval()

    if _text_index is None:
        if not config.TEXT_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Text index not found at {config.TEXT_INDEX_PATH}. "
                "Run scripts/pipeline/embed_memes_fast.py to build it."
            )
        _text_index = faiss.read_index(str(config.TEXT_INDEX_PATH))
        logger.info("Loaded text index with %d vectors", _text_index.ntotal)

    if _image_index is None:
        if config.IMAGE_INDEX_PATH.exists():
            _image_index = faiss.read_index(str(config.IMAGE_INDEX_PATH))
            logger.info("Loaded image index with %d vectors", _image_index.ntotal)
        elif require_image_index:
            raise FileNotFoundError(f"Image index not found at {config.IMAGE_INDEX_PATH}.")
        else:
            logger.warning(
                "Image index missing at %s; only mode='text' will be available",
                config.IMAGE_INDEX_PATH,
            )

    if _id_map is None:
        if not config.ID_MAP_PATH.exists():
            raise FileNotFoundError(f"ID map not found at {config.ID_MAP_PATH}.")
        with open(config.ID_MAP_PATH) as handle:
            _id_map = json.load(handle)
        logger.info("Loaded id map with %d entries", len(_id_map))

    if _collection is None:
        _collection = _open_metadata_store()


def _open_metadata_store():
    """Open the configured metadata store.

    ``mongo`` is the default and keeps ``pymongo`` out of the import path for
    anyone running the ``jsonl`` backend, which needs no database at all.
    """
    backend = config.METADATA_BACKEND
    if backend == "jsonl":
        from services.metadata import JsonlCollection  # noqa: PLC0415

        store = JsonlCollection(config.METADATA_PATH)
        logger.info("Loaded metadata for %d memes from %s", len(store), config.METADATA_PATH)
        return store

    if backend == "mongo":
        from pymongo import MongoClient  # noqa: PLC0415

        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
        return client[config.MONGO_DB][config.MONGO_COLLECTION]

    raise ValueError(f"MEME_METADATA_BACKEND must be 'mongo' or 'jsonl', got {backend!r}")


def available_modes() -> tuple[str, ...]:
    """Modes that can actually be served with the resources currently loaded."""
    if _image_index is None:
        return ("text",)
    return VALID_MODES


def health() -> dict[str, Any]:
    """Readiness snapshot, used by the ``/health`` endpoint."""
    backend = config.METADATA_BACKEND
    store_ok = False
    store_error = None

    if _collection is None:
        store_error = "metadata store not loaded"
    elif backend == "jsonl":
        # Loading the file is the readiness check: JsonlCollection raises on a
        # missing file, so a non-None store is a reachable one.
        store_ok = True
    else:
        try:
            _collection.database.client.admin.command("ping")
            store_ok = True
        except Exception as exc:  # pragma: no cover - depends on live Mongo
            store_error = str(exc)

    return {
        "model_loaded": _model is not None,
        "clip_model": config.CLIP_MODEL,
        "device": config.DEVICE,
        "text_index_vectors": _text_index.ntotal if _text_index is not None else 0,
        "image_index_vectors": _image_index.ntotal if _image_index is not None else 0,
        "id_map_entries": len(_id_map) if _id_map is not None else 0,
        "metadata_backend": backend,
        "metadata_connected": store_ok,
        "metadata_error": store_error,
        # Retained under the old keys so existing clients keep working.
        "mongo_connected": store_ok,
        "mongo_error": store_error,
        "available_modes": list(available_modes()),
    }


def encode_query(query: str):
    """Encode a text query into a unit-norm CLIP embedding of shape (1, dim)."""
    import clip  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if _model is None:
        raise ResourcesNotLoaded("CLIP model is not loaded")

    tokens = clip.tokenize([query], truncate=True).to(config.DEVICE)
    with torch.no_grad():
        embedding = _model.encode_text(tokens).cpu().numpy().astype("float32")

    # FAISS inner product == cosine similarity only for unit-norm vectors.
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding


def _query_index(index, embedding, k: int) -> list[Candidate]:
    """Search one FAISS index and translate vector ids into meme ids."""
    if index is None or index.ntotal == 0:
        return []

    k = min(k, index.ntotal)
    scores, vector_ids = index.search(embedding, k)

    hits: list[Candidate] = []
    for vector_id, score in zip(vector_ids[0], scores[0], strict=False):
        if vector_id < 0:  # FAISS pads short result sets with -1.
            continue
        meme_id = (_id_map or {}).get(str(int(vector_id)))
        if meme_id is None:
            logger.debug("Vector %s has no id_map entry; skipping", vector_id)
            continue
        hits.append((meme_id, float(score)))
    return hits


def _fetch_metadata(meme_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Fetch metadata for several memes in one round trip, keyed by meme_id."""
    if not meme_ids or _collection is None:
        return {}
    cursor = _collection.find({"meme_id": {"$in": list(meme_ids)}}, {"_id": 0})
    return {doc["meme_id"]: doc for doc in cursor}


def _image_url(image_path: str) -> str:
    """Public URL for a meme image served by the API's /images mount."""
    filename = image_path.replace("\\", "/").rsplit("/", 1)[-1]
    return f"{config.PUBLIC_BASE_URL}/images/{filename}"


def search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    mode: str = "hybrid",
    text_weight: float = config.HYBRID_TEXT_WEIGHT,
) -> list[dict[str, Any]]:
    """Retrieve the ``top_k`` memes most similar to ``query``.

    Args:
        query: Natural language query.
        top_k: Number of results to return, capped at ``config.MAX_TOP_K``.
        mode: One of ``text``, ``image`` or ``hybrid``.
        text_weight: Hybrid blend weight for the OCR-text index.

    Returns:
        Result dicts carrying ``meme_id``, ``image_path``, ``image_url``,
        ``ocr_text`` and the fused ``score``, ranked best first.

    Raises:
        ValueError: ``mode`` is unknown, ``top_k`` is not positive, or the
            requested mode needs an index that was not loaded.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not query or not query.strip():
        return []

    top_k = min(top_k, config.MAX_TOP_K)

    if _model is None or _text_index is None or _id_map is None:
        load_resources()

    if mode in ("image", "hybrid") and _image_index is None:
        raise ValueError(
            f"mode={mode!r} needs the image index, which is not loaded. "
            "Build it with scripts/pipeline/embed_memes_fast.py or use mode='text'."
        )

    embedding = encode_query(query)

    if mode == "text":
        candidates = dedupe_candidates(_query_index(_text_index, embedding, top_k))
    elif mode == "image":
        candidates = dedupe_candidates(_query_index(_image_index, embedding, top_k))
    else:
        # Oversample each index so fusion has room to reorder before truncation.
        fetch_k = top_k * config.HYBRID_OVERSAMPLE
        text_hits = _query_index(_text_index, embedding, fetch_k)
        image_hits = _query_index(_image_index, embedding, fetch_k)
        if config.FUSION_STRATEGY == "rrf":
            candidates = fuse_ranks(text_hits, image_hits, text_weight=text_weight, k=config.RRF_K)
        elif config.FUSION_STRATEGY == "sum":
            candidates = fuse_scores(text_hits, image_hits, text_weight=text_weight)
        else:
            raise ValueError(
                f"FUSION_STRATEGY must be 'rrf' or 'sum', got {config.FUSION_STRATEGY!r}"
            )

    candidates = candidates[:top_k]
    metadata = _fetch_metadata([meme_id for meme_id, _ in candidates])

    results: list[dict[str, Any]] = []
    for meme_id, score in candidates:
        doc = metadata.get(meme_id)
        if doc is None:
            # FAISS and Mongo can drift if indexing and metadata loading were
            # run against different dataset snapshots; skip rather than 500.
            logger.warning("No metadata for %s; skipping", meme_id)
            continue
        results.append(
            {
                "meme_id": meme_id,
                "image_path": doc.get("image_path", ""),
                "image_url": _image_url(doc.get("image_path", "")),
                "ocr_text": doc.get("ocr_text", ""),
                "score": round(float(score), 4),
            }
        )
    return results


def search_text(query: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Backwards-compatible OCR-text-only search (pre-hybrid API)."""
    return search(query, top_k=top_k, mode="text")
