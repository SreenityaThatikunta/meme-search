import streamlit as st
import requests
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

API_URL = "http://127.0.0.1:8000/search/text"

st.set_page_config(page_title="Meme Search", layout="wide")

st.title("🧠 Meme Search Engine")
st.caption("CLIP + FAISS powered semantic meme search")

# -----------------------------
# QUERY INPUT
# -----------------------------
query = st.text_input("Enter a text query", placeholder="spiderman, relatable, exams, etc.")
top_k = st.slider("Number of results", 1, 20, 5)

if st.button("Search") and query.strip():
    with st.spinner("Searching memes..."):
        resp = requests.post(
            API_URL,
            json={"query": query, "top_k": top_k},
            timeout=10
        )

    if resp.status_code != 200:
        st.error(f"API Error: {resp.text}")
    else:
        results = resp.json()

        if len(results) == 0:
            st.warning("No memes found 😕")
        else:
            cols = st.columns(3)

            for i, meme in enumerate(results):
                col = cols[i % 3]

                with col:
                    # try:
                    #     # st.write("DEBUG image_path:", meme["image_path"])
                    #     img_path = os.path.join(PROJECT_ROOT, meme["image_path"])
                    #     if not os.path.exists(img_path):
                    #         st.error(f"File not found: {img_path}")
                    #         continue
                    #     img = Image.open(img_path)
                    #     st.image(img, use_container_width=True)
                    # except Exception:
                    #     st.write("DEBUG PROJECT_ROOT:", PROJECT_ROOT)
                    #     st.write("DEBUG image_path:", meme["image_path"])
                    #     st.write("DEBUG resolved path:", img_path)
                    #     st.write("DEBUG exists?:", os.path.exists(img_path))
                    #     st.error("Image not found")

                    try:
                        img_path = os.path.join(PROJECT_ROOT, meme["image_path"])
                        img = Image.open(img_path).convert("RGB")
                        st.image(img, use_column_width=True)
                    except Exception as e:
                        st.error(f"Failed to load image: {e}")

                    

                    st.markdown(f"**Score:** `{meme['score']:.3f}`")

                    if meme.get("text"):
                        st.caption(meme["text"])