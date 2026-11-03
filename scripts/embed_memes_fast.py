"""Build the CLIP embedding indexes.

Produces two FAISS inner-product indexes over L2-normalised 512-d CLIP vectors:

* ``image.index`` - one vector per meme image.
* ``text.index``  - one vector per meme's OCR text.

Both share a single ``vector_id`` space, so one ``id_map.json`` resolves a hit
from either index back to a ``meme_id``.
"""

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clip  # noqa: E402

import config  # noqa: E402 - needs the project root on sys.path first

# Some memes in public dumps are truncated; load what is there rather than fail.
ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = int(os.getenv("MEME_BATCH_SIZE", "32"))
# Thumbnail before preprocessing: CLIP resizes to 224px anyway, and this cuts
# decode time substantially on large meme images.
THUMBNAIL_SIZE = (512, 512)
IN_FILE = config.DATA_DIR / "memes_ocr.jsonl"


def main() -> None:
    if not IN_FILE.exists():
        raise SystemExit(f"{IN_FILE} not found. Run scripts/run_ocr_fast.py first.")

    device = config.DEVICE
    print(f"[INFO] Loading {config.CLIP_MODEL} on {device}")
    model, preprocess = clip.load(config.CLIP_MODEL, device=device)
    model.eval()

    def load_image(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE)
        return preprocess(img)

    with open(IN_FILE) as handle:
        records = [json.loads(line) for line in handle]

    image_batches, text_batches = [], []
    id_map: dict[int, str] = {}
    vector_id = 0

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc="embedding"):
        batch = records[start : start + BATCH_SIZE]

        images = torch.stack([load_image(r["image_path"]) for r in batch]).to(device)
        with torch.no_grad():
            image_emb = model.encode_image(images).cpu().numpy().astype("float32")
        image_emb /= np.linalg.norm(image_emb, axis=1, keepdims=True)

        # Empty OCR text would give a degenerate embedding, so fall back to a
        # neutral token that at least places the meme in the right region.
        texts = [r["text"]["ocr"] or "meme" for r in batch]
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(tokens).cpu().numpy().astype("float32")
        text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)

        image_batches.append(image_emb)
        text_batches.append(text_emb)

        for record in batch:
            id_map[vector_id] = record["meme_id"]
            record["vector_id"] = vector_id
            vector_id += 1

    # Inner product over unit-norm vectors == cosine similarity.
    image_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    text_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    image_index.add(np.vstack(image_batches))
    text_index.add(np.vstack(text_batches))

    faiss.write_index(image_index, str(config.IMAGE_INDEX_PATH))
    faiss.write_index(text_index, str(config.TEXT_INDEX_PATH))

    with open(config.ID_MAP_PATH, "w") as handle:
        json.dump(id_map, handle)

    with open(config.METADATA_PATH, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(
        f"[DONE] Indexed {vector_id} memes "
        f"-> {config.TEXT_INDEX_PATH.name}, {config.IMAGE_INDEX_PATH.name}"
    )


if __name__ == "__main__":
    main()
