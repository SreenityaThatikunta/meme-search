"""Streamlit front end for meme search.

Runs in one of two modes, set by ``MEME_SEARCH_MODE``:

``http`` (default)
    Call the FastAPI service over HTTP. This is the local development setup —
    the API is a separate process and this UI is one of its two clients.
``local``
    Import the search service and call it in-process, with no API and no
    database (pair it with ``MEME_METADATA_BACKEND=jsonl``). This is what the
    hosted demo runs, where a single process has to serve everything.
"""

import os
from pathlib import Path

# CLIP and FAISS both link against OpenMP; on macOS the duplicate-runtime check
# aborts the process (SIGABRT) before Streamlit can report anything useful. In
# `local` mode this UI loads both in-process, so it needs the same escape hatch
# main.py sets. Must happen before torch or faiss is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

# config.py imports only os and pathlib, so it is safe to load in either mode
# and keeps defaults single-sourced.
import config

SEARCH_MODE = os.getenv("MEME_SEARCH_MODE", "http")
API_BASE_URL = os.getenv("MEME_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SEARCH_URL = f"{API_BASE_URL}/search/text"
HEALTH_URL = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT = 30
COLUMNS = 3

MODE_HELP = {
    "hybrid": "Blend both signals — best general-purpose ranking.",
    "text": "Match the OCR text scraped off each meme.",
    "image": "Match what the meme looks like, ignoring its caption.",
}

st.set_page_config(page_title="Meme Search", layout="wide")
st.title("🧠 Meme Search Engine")
st.caption("Semantic meme retrieval with CLIP embeddings and FAISS")


@st.cache_resource
def _local_service():
    """Load the search service once per process, at startup rather than per query.

    CLIP is ~28 ms per query but several seconds to load, so this pays that cost
    while the page is first rendering instead of inside the first search.
    """
    from services import search as search_service

    search_service.load_resources()
    return search_service


def fetch_health() -> dict:
    """Report which resources loaded, so the UI only offers valid modes."""
    if SEARCH_MODE == "local":
        try:
            return _local_service().health()
        except Exception as exc:
            return {"error": str(exc)}

    import requests  # noqa: PLC0415 - only the http mode needs it

    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def run_search(query: str, top_k: int, mode: str, text_weight: float) -> tuple[list, str | None]:
    """Run one search. Returns ``(results, error_message)``."""
    if SEARCH_MODE == "local":
        try:
            results = _local_service().search(
                query, top_k=top_k, mode=mode, text_weight=text_weight
            )
            return results, None
        except ValueError as exc:
            return [], str(exc)
        except Exception as exc:
            return [], f"Search failed: {exc}"

    import requests  # noqa: PLC0415

    try:
        response = requests.post(
            SEARCH_URL,
            json={"query": query, "top_k": top_k, "mode": mode, "text_weight": text_weight},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return [], f"Could not reach the API: {exc}"

    if response.status_code != 200:
        return [], f"API error {response.status_code}: {response.text}"
    return response.json(), None


def image_source(meme: dict) -> str:
    """Where to load a result's image from.

    In ``http`` mode the API serves images at ``/images`` and hands back a URL.
    In ``local`` mode there is no such mount, so read the file directly —
    rebuilding the path from ``MEME_IMAGE_DIR`` rather than trusting the
    ``image_path`` baked into the metadata, which was recorded relative to
    wherever the indexes were built.
    """
    if SEARCH_MODE == "local":
        return str(config.IMAGE_DIR / Path(meme.get("image_path", "")).name)
    return meme["image_url"]


health = fetch_health()

with st.sidebar:
    st.header("Backend")
    if "error" in health:
        if SEARCH_MODE == "local":
            st.error("Search service failed to load")
        else:
            st.error(f"API unreachable at {API_BASE_URL}")
            st.code("uvicorn main:app --port 8000", language="bash")
        st.caption(health["error"])
    else:
        st.success(
            "Running in-process" if SEARCH_MODE == "local" else f"Connected to {API_BASE_URL}"
        )
        st.metric("Indexed memes", f"{health.get('text_index_vectors', 0):,}")
        st.caption(f"Model: {health.get('clip_model')} on {health.get('device')}")
        if not health.get("metadata_connected", health.get("mongo_connected", True)):
            backend = health.get("metadata_backend", "metadata store")
            st.warning(f"{backend} is not reachable; results cannot be hydrated.")

available_modes = health.get("available_modes", ["hybrid", "text", "image"])
# The service reports modes in (text, image, hybrid) order, but hybrid is the
# documented default and by far the strongest (Recall@10 0.599 vs 0.351 for
# text), so it should be what the dropdown offers first.
available_modes = sorted(available_modes, key=lambda m: m != "hybrid")

left, middle, right = st.columns([3, 1, 1])
with left:
    query = st.text_input("Search query", placeholder="spiderman, exam stress, monday morning...")
with middle:
    mode = st.selectbox("Mode", available_modes, help="How results are ranked")
with right:
    top_k = st.slider("Results", 1, 20, 6)

text_weight = config.HYBRID_TEXT_WEIGHT
if mode == "hybrid":
    # Default comes from config (0.2), which is the weight measured as optimal on
    # the known-item eval set — not a midpoint guess. See docs/EVALUATION.md.
    text_weight = st.slider(
        "OCR text weight",
        0.0,
        1.0,
        config.HYBRID_TEXT_WEIGHT,
        0.05,
        help="0 = rank on appearance only, 1 = rank on caption text only",
    )

st.caption(MODE_HELP.get(mode, ""))

if os.getenv("MEME_DEMO_SUBSET") == "1":
    st.info(
        f"Demo build: searching a {health.get('text_index_vectors', 0):,}-meme subset. "
        "The published Recall@10 of 0.599 and ~29 ms latency were measured on the full "
        "6,989-meme dataset — retrieval behaviour is the same, there is just less to find.",
        icon="ℹ️",
    )

if st.button("Search", type="primary") and query.strip():
    with st.spinner("Searching memes..."):
        results, error = run_search(query, top_k, mode, text_weight)

    if error:
        st.error(error)
        st.stop()

    if not results:
        st.warning("No memes matched that query.")
        st.stop()

    columns = st.columns(COLUMNS)
    for position, meme in enumerate(results):
        with columns[position % COLUMNS]:
            st.image(image_source(meme), use_container_width=True)
            st.markdown(f"**Score:** `{meme['score']:.3f}`")
            if meme.get("ocr_text"):
                st.caption(meme["ocr_text"][:140])
