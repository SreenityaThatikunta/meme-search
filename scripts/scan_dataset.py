"""Scan the meme image directory into a JSONL manifest."""

import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402 - needs the project root on sys.path first

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
OUT_FILE = config.DATA_DIR / "memes.jsonl"


def main() -> None:
    if not config.IMAGE_DIR.is_dir():
        raise SystemExit(f"Image directory {config.IMAGE_DIR} not found.")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    filenames = sorted(
        name for name in config.IMAGE_DIR.iterdir() if name.suffix.lower() in IMAGE_SUFFIXES
    )

    records = []
    skipped = 0
    for idx, path in enumerate(tqdm(filenames, desc="scanning")):
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception as exc:
            # A handful of files in public meme dumps are truncated or mislabelled.
            print(f"[WARN] Skipping unreadable {path.name}: {exc}")
            skipped += 1
            continue

        records.append(
            {
                "meme_id": f"meme_{idx:06d}",
                "image_path": str(path),
                "width": width,
                "height": height,
            }
        )

    with open(OUT_FILE, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"Saved {len(records)} memes to {OUT_FILE} ({skipped} skipped)")


if __name__ == "__main__":
    main()
