"""Draw a blind sample of pooled pairs for a second annotator to judge.

The judgments in data/relevance_pool.jsonl were made by a single AI annotator,
so there is no agreement figure and no human check on them. This cuts a sample
small enough to judge in one sitting and writes it as a CSV with an **empty**
grade column: the existing grades are deliberately withheld, so the second
annotator is not anchored by them.

Sampling is stratified by query (an equal number from each) so every query type
is represented rather than the sample landing on whichever queries pooled most
candidates. Items stay grouped by query, since a relevance grade only means
anything against the query it was judged for.

Workflow:
    1. python scripts/build_judgment_sample.py
    2. python scripts/build_contact_sheets.py --pool data/judgment_sample.jsonl --out sheets/
    3. Fill the `relevance` column in data/judgment_sample.csv with 0, 1 or 2:
         0 = irrelevant, 1 = partially relevant, 2 = relevant
    4. python scripts/compare_judgments.py
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402 - needs the project root on sys.path first

POOL_PATH = config.DATA_DIR / "relevance_pool.jsonl"
SAMPLE_CSV = config.DATA_DIR / "judgment_sample.csv"
SAMPLE_JSONL = config.DATA_DIR / "judgment_sample.jsonl"
PER_QUERY = 3
SEED = 11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=int, default=PER_QUERY)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not POOL_PATH.exists():
        raise SystemExit(f"{POOL_PATH} not found. Run build_relevance_pool.py first.")

    by_query: dict[str, list[dict]] = defaultdict(list)
    with open(POOL_PATH) as handle:
        for line in handle:
            row = json.loads(line)
            by_query[row["query"]].append(row)

    rng = random.Random(args.seed)
    sampled = []
    for query in sorted(by_query):
        rows = by_query[query]
        take = min(args.per_query, len(rows))
        sampled.extend(rng.sample(rows, take))

    # Deliberately NOT shuffled across queries: a relevance judgment is only
    # meaningful against its query, so both the CSV and the contact sheets are
    # grouped by query. The protection against anchoring is that the first-pass
    # grades are withheld, not that the order is scrambled. Within a query the
    # order is already shuffled, from the pool build.
    sampled.sort(key=lambda row: row["query"])

    with open(SAMPLE_JSONL, "w") as handle:
        for position, row in enumerate(sampled, start=1):
            handle.write(
                json.dumps(
                    {
                        "item": position,
                        "query": row["query"],
                        "meme_id": row["meme_id"],
                        "image_path": row["image_path"],
                        "ocr_text": row["ocr_text"],
                        "relevance": None,
                    }
                )
                + "\n"
            )

    with open(SAMPLE_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "query", "meme_id", "relevance", "ocr_text"])
        for position, row in enumerate(sampled, start=1):
            text = " ".join(row["ocr_text"].split())[:90]
            writer.writerow([position, row["query"], row["meme_id"], "", text])

    grades = [row["relevance"] for row in sampled]
    print(f"sampled {len(sampled)} pairs from {len(by_query)} queries")
    print(
        f"  (withheld first-pass grade mix: {{0: {grades.count(0)}, 1: {grades.count(1)}, 2: {grades.count(2)}}})"
    )
    print(f"wrote {SAMPLE_CSV}   <- fill the 'relevance' column")
    print(f"wrote {SAMPLE_JSONL} <- for build_contact_sheets.py")


if __name__ == "__main__":
    main()
