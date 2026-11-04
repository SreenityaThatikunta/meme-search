"""Build a known-item retrieval evaluation set.

Ground truth comes from the source dataset's *human-corrected* captions in
``data/labels.csv`` — a column this project never indexes. The search indexes
embed our own EasyOCR output instead, and the two disagree for ~93% of memes
(real OCR errors: ``1oyears`` for ``10 years``, dropped and reordered tokens).
A query drawn from the corrected caption is therefore not a lookup of the
indexed string.

Each query is a short contiguous window of the corrected caption rather than the
whole thing, which models what a user actually remembers ("one does not simply")
instead of a full transcript.

What this measures, and what it does not:

* It measures **known-item retrieval** — "find the specific meme whose caption
  went roughly like this". That is a real and probably dominant meme-search task.
* It does **not** measure general semantic relevance ("something relatable about
  Mondays"), because there is exactly one correct answer per query.
* It **structurally favours** ``mode="text"``, since the query derives from the
  caption that the text index embeds. See the warning in evaluate_retrieval.py:
  ``text_weight`` must not be tuned to maximise a score on this set.

Usage:
    python scripts/eval/build_eval_set.py [--queries 1000] [--window 6] [--seed 42]
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402 - needs the project root on sys.path first

LABELS_PATH = config.DATA_DIR / "labels.csv"
OUT_PATH = config.DATA_DIR / "eval_queries.jsonl"
# A window needs enough surrounding caption to be a fragment rather than the whole thing.
MIN_CAPTION_WORDS = 10


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def load_metadata() -> dict[str, str]:
    """Map image basename -> meme_id using the indexed metadata."""
    if not config.METADATA_PATH.exists():
        raise SystemExit(f"{config.METADATA_PATH} not found. Run the indexing pipeline first.")
    mapping = {}
    with open(config.METADATA_PATH) as handle:
        for line in handle:
            record = json.loads(line)
            basename = record["image_path"].replace("\\", "/").rsplit("/", 1)[-1]
            mapping[basename] = record["meme_id"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--window", type=int, default=6, help="words per query")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        raise SystemExit(
            f"{LABELS_PATH} not found. It ships with the source dataset rather than this "
            "repo (see 'Add the dataset' in the README). You only need it to rebuild the "
            "query set — data/eval_queries.jsonl is committed, so scripts/eval/"
            "evaluate_retrieval.py reproduces the published numbers without it."
        )

    basename_to_meme = load_metadata()
    rng = random.Random(args.seed)

    candidates = []
    with open(LABELS_PATH) as handle:
        for row in csv.DictReader(handle):
            meme_id = basename_to_meme.get(row["image_name"])
            if meme_id is None:
                continue
            caption = normalise(row["text_corrected"])
            words = caption.split()
            if len(words) < max(MIN_CAPTION_WORDS, args.window):
                continue
            candidates.append((meme_id, words, caption))

    print(f"eligible memes (corrected caption >= {MIN_CAPTION_WORDS} words): {len(candidates)}")
    if len(candidates) < args.queries:
        print(f"[WARN] only {len(candidates)} eligible; using all of them")

    sampled = rng.sample(candidates, min(args.queries, len(candidates)))

    with open(OUT_PATH, "w") as handle:
        for meme_id, words, caption in sampled:
            start = rng.randint(0, len(words) - args.window)
            query = " ".join(words[start : start + args.window])
            handle.write(
                json.dumps(
                    {
                        "query": query,
                        "gold_meme_id": meme_id,
                        "source_caption": caption,
                    }
                )
                + "\n"
            )

    print(f"wrote {len(sampled)} queries to {OUT_PATH} (window={args.window}, seed={args.seed})")


if __name__ == "__main__":
    main()
