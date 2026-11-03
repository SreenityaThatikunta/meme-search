"""Streamlit front end for the meme search API."""

import os

import requests
import streamlit as st

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


@st.cache_data(ttl=30)
def fetch_health() -> dict:
    """Ask the API which resources loaded, so the UI can offer valid modes."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


health = fetch_health()

with st.sidebar:
    st.header("Backend")
    if "error" in health:
        st.error(f"API unreachable at {API_BASE_URL}")
        st.caption(health["error"])
        st.code("uvicorn main:app --port 8000", language="bash")
    else:
        st.success(f"Connected to {API_BASE_URL}")
        st.metric("Indexed memes", f"{health.get('text_index_vectors', 0):,}")
        st.caption(f"Model: {health.get('clip_model')} on {health.get('device')}")
        if not health.get("mongo_connected"):
            st.warning("MongoDB is not reachable; results cannot be hydrated.")

available_modes = health.get("available_modes", ["hybrid", "text", "image"])

left, middle, right = st.columns([3, 1, 1])
with left:
    query = st.text_input("Search query", placeholder="spiderman, exam stress, monday morning...")
with middle:
    mode = st.selectbox("Mode", available_modes, help="How results are ranked")
with right:
    top_k = st.slider("Results", 1, 20, 6)

text_weight = 0.5
if mode == "hybrid":
    text_weight = st.slider(
        "OCR text weight",
        0.0,
        1.0,
        0.5,
        0.05,
        help="0 = rank on appearance only, 1 = rank on caption text only",
    )

st.caption(MODE_HELP.get(mode, ""))

if st.button("Search", type="primary") and query.strip():
    with st.spinner("Searching memes..."):
        try:
            response = requests.post(
                SEARCH_URL,
                json={
                    "query": query,
                    "top_k": top_k,
                    "mode": mode,
                    "text_weight": text_weight,
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            st.stop()

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        st.stop()

    results = response.json()
    if not results:
        st.warning("No memes matched that query.")
        st.stop()

    columns = st.columns(COLUMNS)
    for position, meme in enumerate(results):
        with columns[position % COLUMNS]:
            # Images come from the API's /images mount, so the UI needs no
            # access to the dataset on disk.
            st.image(meme["image_url"], use_container_width=True)
            st.markdown(f"**Score:** `{meme['score']:.3f}`")
            if meme.get("ocr_text"):
                st.caption(meme["ocr_text"][:140])
