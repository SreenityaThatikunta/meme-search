import faiss
import json
import torch
import clip
import numpy as np
from pymongo import MongoClient

DEVICE = "cpu"

model = None
text_index = None
id_map = None
db = None

def load_resources():
    global model, text_index, id_map, db

    if model is None:
        model, _ = clip.load("ViT-B/32", device=DEVICE)
        model.eval()

    if text_index is None:
        text_index = faiss.read_index("./data/text.index")

    if id_map is None:
        with open("data/id_map.json") as f:
            id_map = json.load(f)

    if db is None:
        db = MongoClient()["meme_search"]["memes"]


def search_text(query, top_k=5):
    if model is None:
        load_resources()

    tokens = clip.tokenize([query], truncate=True).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_text(tokens).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    scores, ids = text_index.search(emb, top_k)

    results = []

    for vid, score in zip(ids[0], scores[0]):
        meme_id = id_map.get(str(vid))

        if meme_id is None:
            continue

        doc = db.find_one({"meme_id": meme_id}, {"_id": 0})

        if doc is None:
            # Mongo + FAISS mismatch → skip safely
            continue

        # NEVER mutate original doc in-place (good practice)
        result = {
            **doc,
            "score": float(score)
        }

        results.append(result)

    return results