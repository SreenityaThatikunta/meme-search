import os
import json
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "data/images/images"
OUT_FILE = "data/memes.jsonl"

os.makedirs("data", exist_ok=True)

records = []
for idx, fname in enumerate(sorted(os.listdir(IMAGE_DIR))):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        continue

    path = os.path.join(IMAGE_DIR, fname)
    with Image.open(path) as img:
        w, h = img.size

    records.append({
        "meme_id": f"meme_{idx:06d}",
        "image_path": path,
        "width": w,
        "height": h
    })

with open(OUT_FILE, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Saved {len(records)} memes")