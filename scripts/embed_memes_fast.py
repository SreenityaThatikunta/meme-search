import json
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------
BATCH_SIZE = 32
DEVICE = "mps"  # stable on macOS
IN_FILE = "data/memes_ocr.jsonl"
DATA_DIR = "data"

# ----------------------------------------

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

image_index = faiss.IndexFlatIP(512)
text_index = faiss.IndexFlatIP(512)

image_embs = []
text_embs = []
id_map = {}

with open(IN_FILE) as f:
    records = [json.loads(l) for l in f]

from PIL import ImageFile

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((512, 512))
    return preprocess(img)

vector_id = 0

for i in tqdm(range(0, len(records), BATCH_SIZE)):
    batch = records[i:i+BATCH_SIZE]

    # ---- IMAGE EMBEDDINGS ----
    imgs = torch.stack([load_image(r["image_path"]) for r in batch]).to(DEVICE)
    with torch.no_grad():
        img_emb = model.encode_image(imgs).cpu().numpy()
    img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)

    # ---- TEXT EMBEDDINGS ----
    texts = [r["text"]["ocr"] or "meme" for r in batch]
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        txt_emb = model.encode_text(tokens).cpu().numpy()
    txt_emb /= np.linalg.norm(txt_emb, axis=1, keepdims=True)

    image_embs.append(img_emb)
    text_embs.append(txt_emb)

    for r in batch:
        id_map[vector_id] = r["meme_id"]
        r["vector_id"] = vector_id
        vector_id += 1

# ---- ADD TO FAISS ONCE ----
image_index.add(np.vstack(image_embs))
text_index.add(np.vstack(text_embs))

faiss.write_index(image_index, os.path.join(DATA_DIR, "image.index"))
faiss.write_index(text_index, os.path.join(DATA_DIR, "text.index"))

with open(os.path.join(DATA_DIR, "id_map.json"), "w") as f:
    json.dump(id_map, f)

with open("data/memes_final.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print("Fast embedding + indexing complete")