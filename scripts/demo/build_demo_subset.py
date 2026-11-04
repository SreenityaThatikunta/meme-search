"""Build a small, self-contained dataset for the hosted demo.

The full dataset is 717 MB of images, which is more than a free Space wants to
hold. This carves out a few hundred memes that keep the demo honest:

1. Every meme judged relevant for one of the 16 documented relevance queries,
   so ``exam stress`` and ``monday morning`` still return their known-good
   results rather than nothing.
2. A seeded random top-up of memes whose OCR recovered at least a few words, so
   ``text`` mode has material to match.

**No re-embedding.** ``IndexFlatIP`` stores its vectors verbatim, so
``reconstruct_n`` lifts them straight out of the committed indexes and a new
index takes the rows we want. This needs neither CLIP nor a GPU — only faiss and
the image files for the chosen memes.

The invariant that matters (see docs/ARCHITECTURE.md): row *i* of text.index and
row *i* of image.index must describe the same meme. Both subset indexes are
therefore built from one ordered meme list, and id_map.json is written from that
same list, with vector_ids renumbered to 0..N-1.

Usage:
    python scripts/demo/build_demo_subset.py [--size 400] [--seed 42] [--out demo/data]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402 - needs the project root on sys.path first

MIN_OCR_WORDS = 3


def load_metadata() -> dict[str, dict]:
    """Read memes_final.jsonl, keyed on meme_id."""
    if not config.METADATA_PATH.exists():
        raise SystemExit(
            f"{config.METADATA_PATH} not found. Run scripts/pipeline/embed_memes_fast.py first."
        )
    records = {}
    with open(config.METADATA_PATH) as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["meme_id"]] = record
    return records


def judged_relevant_ids() -> list[str]:
    """Meme ids graded relevant for the documented relevance queries."""
    pool_path = config.DATA_DIR / "relevance_pool.jsonl"
    if not pool_path.exists():
        print(f"  note: {pool_path} not found; demo queries may return little")
        return []
    ids = []
    with open(pool_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("relevance", 0) >= 1:
                ids.append(entry["meme_id"])
    return ids


def select_memes(records: dict[str, dict], size: int, seed: int) -> list[str]:
    """Choose which memes the demo serves, as an ordered, de-duplicated list."""
    rng = np.random.default_rng(seed)

    selected: list[str] = []
    seen: set[str] = set()

    # Priority 1: anything a human graded relevant, so documented queries work.
    for meme_id in judged_relevant_ids():
        if meme_id in records and meme_id not in seen:
            selected.append(meme_id)
            seen.add(meme_id)
    print(f"  {len(selected)} memes from the judged relevance pool")

    # Priority 2: top up with text-bearing memes, seeded for reproducibility.
    candidates = sorted(
        meme_id
        for meme_id, record in records.items()
        if meme_id not in seen
        and len((record.get("text") or {}).get("ocr", "").split()) >= MIN_OCR_WORDS
    )
    shortfall = max(0, size - len(selected))
    if shortfall and candidates:
        picked = rng.choice(candidates, size=min(shortfall, len(candidates)), replace=False)
        topped_up = [str(meme_id) for meme_id in picked]
        selected.extend(topped_up)
        print(f"  {len(topped_up)} memes topped up from {len(candidates)} text-bearing candidates")

    return selected[:size]


def slice_index(source_path: Path, vector_ids: list[int], dim: int) -> faiss.IndexFlatIP:
    """Build a new IndexFlatIP holding only ``vector_ids``, in that order."""
    source = faiss.read_index(str(source_path))
    if source.d != dim:
        raise SystemExit(f"{source_path} has dim {source.d}, expected {dim}")

    vectors = np.vstack([source.reconstruct(int(vid)).reshape(1, -1) for vid in vector_ids])
    target = faiss.IndexFlatIP(dim)
    target.add(vectors.astype("float32"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", type=int, default=400, help="memes to include")
    parser.add_argument("--seed", type=int, default=42, help="top-up sampling seed")
    parser.add_argument("--out", type=Path, default=Path("demo/data"), help="output directory")
    args = parser.parse_args()

    out_dir = (config.PROJECT_ROOT / args.out).resolve()
    images_out = out_dir / "images" / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {config.METADATA_PATH}")
    records = load_metadata()
    print(f"Selecting up to {args.size} memes from {len(records)}:")
    selected = select_memes(records, args.size, args.seed)
    if not selected:
        raise SystemExit("No memes selected; nothing to build.")

    # One ordered list drives everything below, which is what keeps the two
    # indexes and the id map consistent with each other.
    original_vector_ids = [records[meme_id]["vector_id"] for meme_id in selected]

    print(f"Slicing indexes to {len(selected)} vectors (no re-embedding)")
    text_index = slice_index(config.TEXT_INDEX_PATH, original_vector_ids, config.EMBEDDING_DIM)
    image_index = slice_index(config.IMAGE_INDEX_PATH, original_vector_ids, config.EMBEDDING_DIM)
    faiss.write_index(text_index, str(out_dir / "text.index"))
    faiss.write_index(image_index, str(out_dir / "image.index"))

    id_map = {str(row): meme_id for row, meme_id in enumerate(selected)}
    (out_dir / "id_map.json").write_text(json.dumps(id_map))

    missing_images = []
    with open(out_dir / "memes_final.jsonl", "w") as handle:
        for row, meme_id in enumerate(selected):
            record = dict(records[meme_id])
            record["vector_id"] = row  # renumbered to match the sliced indexes
            handle.write(json.dumps(record) + "\n")

            filename = Path(record["image_path"]).name
            source_image = config.IMAGE_DIR / filename
            if source_image.exists():
                shutil.copy2(source_image, images_out / filename)
            else:
                missing_images.append(filename)

    payload_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"\nWrote {out_dir}")
    print(f"  {len(selected)} memes, {text_index.ntotal} vectors per index, {payload_mb:.1f} MB")
    if missing_images:
        print(f"  WARNING: {len(missing_images)} image(s) not found, e.g. {missing_images[:3]}")
        print(f"  Check MEME_IMAGE_DIR (currently {config.IMAGE_DIR})")


if __name__ == "__main__":
    main()
