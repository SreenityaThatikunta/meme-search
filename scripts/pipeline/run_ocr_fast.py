"""Extract meme text with EasyOCR.

EasyOCR is the slow step of the pipeline, so two optimisations apply:

1. A cheap Canny edge-density heuristic skips images that almost certainly carry
   no text, avoiding a full OCR pass on them.
2. The remaining work is spread across a process pool.
"""

import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import easyocr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402 - needs the project root on sys.path first

IN_FILE = config.DATA_DIR / "memes.jsonl"
OUT_FILE = config.DATA_DIR / "memes_ocr.jsonl"

# Mean Canny edge response below this is treated as "no legible text".
EDGE_THRESHOLD = 5

_reader = None


def get_reader() -> easyocr.Reader:
    """Build the OCR reader once per worker process."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def likely_has_text(image_path: str) -> bool:
    """Cheap pre-filter: does this image have enough edge detail to hold text?"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img = cv2.resize(img, (256, 256))
    return cv2.Canny(img, 50, 150).mean() > EDGE_THRESHOLD


def ocr_one(record: dict) -> dict:
    """Attach ``record["text"]["ocr"]``, empty when nothing readable is found."""
    try:
        if not likely_has_text(record["image_path"]):
            record["text"] = {"ocr": ""}
            return record

        lines = get_reader().readtext(record["image_path"], detail=0)
        record["text"] = {"ocr": " ".join(lines).lower()}
    except Exception as exc:
        # One unreadable meme should not abort a 7k-image run.
        print(f"[WARN] OCR failed for {record.get('image_path')}: {exc}")
        record["text"] = {"ocr": ""}
    return record


def main() -> None:
    if not IN_FILE.exists():
        raise SystemExit(f"{IN_FILE} not found. Run scripts/pipeline/scan_dataset.py first.")

    with open(IN_FILE) as handle:
        records = [json.loads(line) for line in handle]

    workers = max(cpu_count() - 1, 1)
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(ocr_one, records), total=len(records), desc="ocr"))

    with open(OUT_FILE, "w") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")

    with_text = sum(1 for r in results if r["text"]["ocr"])
    print(f"[DONE] OCR complete: {with_text}/{len(results)} memes have text")


if __name__ == "__main__":
    main()
