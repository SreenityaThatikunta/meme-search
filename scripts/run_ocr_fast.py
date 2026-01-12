import json
import cv2
import easyocr
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

IN_FILE = "data/memes.jsonl"
OUT_FILE = "data/memes_ocr.jsonl"

reader = easyocr.Reader(["en"], gpu=False)

def has_text_fast(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img = cv2.resize(img, (256, 256))
    edges = cv2.Canny(img, 50, 150)
    return edges.mean() > 5  # cheap heuristic

def ocr_one(record):
    try:
        if not has_text_fast(record["image_path"]):
            record["text"] = {"ocr": ""}
            return record

        text = reader.readtext(record["image_path"], detail=0)
        record["text"] = {"ocr": " ".join(text).lower()}
        return record

    except Exception:
        record["text"] = {"ocr": ""}
        return record

if __name__ == "__main__":
    with open(IN_FILE) as f:
        records = [json.loads(l) for l in f]

    with Pool(max(cpu_count() - 1, 1)) as pool:
        results = list(tqdm(pool.imap(ocr_one, records), total=len(records)))

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("Fast OCR complete")