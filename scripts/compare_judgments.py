"""Compare a second annotator's grades against the first-pass judgments.

Reports raw agreement, Cohen's kappa (plain and quadratic-weighted), and a
confusion matrix, then lists the items the two annotators disagreed on most so
the disagreement can be inspected rather than just quantified.

Needs no model, index or database.

Usage:
    python scripts/build_judgment_sample.py      # writes the blank CSV
    # ... fill in the relevance column ...
    python scripts/compare_judgments.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402 - needs the project root on sys.path first
from services.metrics import cohen_kappa  # noqa: E402

POOL_PATH = config.DATA_DIR / "relevance_pool.jsonl"
SAMPLE_CSV = config.DATA_DIR / "judgment_sample.csv"
GRADES = (0, 1, 2)
LABELS = {0: "irrelevant", 1: "partial", 2: "relevant"}


def interpret(kappa: float) -> str:
    """Conventional reading of a kappa value."""
    if kappa < 0:
        return "worse than chance"
    if kappa < 0.2:
        return "poor"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "near-perfect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=str(SAMPLE_CSV))
    args = parser.parse_args()

    sample_path = Path(args.sample)
    for path in (POOL_PATH, sample_path):
        if not path.exists():
            raise SystemExit(f"{path} not found.")

    first_pass = {}
    with open(POOL_PATH) as handle:
        for line in handle:
            row = json.loads(line)
            first_pass[(row["query"], row["meme_id"])] = row["relevance"]

    mine, theirs, items, blank = [], [], [], 0
    with open(sample_path, newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("relevance") or "").strip()
            if raw == "":
                blank += 1
                continue
            try:
                grade = int(raw)
            except ValueError:
                raise SystemExit(
                    f"item {row['item']}: relevance must be 0, 1 or 2, got {raw!r}"
                ) from None
            if grade not in GRADES:
                raise SystemExit(f"item {row['item']}: {grade} is not 0, 1 or 2")

            key = (row["query"], row["meme_id"])
            if key not in first_pass:
                raise SystemExit(f"item {row['item']}: {key} is not in the pool")

            theirs.append(grade)
            mine.append(first_pass[key])
            items.append(row)

    if blank:
        print(f"[note] {blank} of {blank + len(theirs)} items are still unjudged\n")
    if not theirs:
        raise SystemExit(
            f"No grades found. Fill the 'relevance' column in {sample_path} with 0, 1 or 2."
        )

    exact = sum(a == b for a, b in zip(mine, theirs, strict=True))
    adjacent = sum(abs(a - b) <= 1 for a, b in zip(mine, theirs, strict=True))
    plain = cohen_kappa(mine, theirs)
    weighted = cohen_kappa(mine, theirs, weights="quadratic")

    print(f"items compared: {len(theirs)}")
    print(f"  exact agreement      {exact / len(theirs):6.1%}  ({exact}/{len(theirs)})")
    print(f"  within one grade     {adjacent / len(theirs):6.1%}")
    print(f"  Cohen's kappa        {plain:6.3f}  ({interpret(plain)})")
    print(f"  quadratic-weighted   {weighted:6.3f}  ({interpret(weighted)})")

    print("\nconfusion matrix (rows: first pass, columns: second annotator)")
    header = "            " + "".join(f"{LABELS[g]:>12}" for g in GRADES)
    print(header)
    for row_grade in GRADES:
        cells = "".join(
            f"{sum(1 for a, b in zip(mine, theirs, strict=True) if a == row_grade and b == col_grade):>12}"
            for col_grade in GRADES
        )
        print(f"{LABELS[row_grade]:>12}{cells}")

    disagreements = [
        (abs(a - b), a, b, row) for a, b, row in zip(mine, theirs, items, strict=True) if a != b
    ]
    if disagreements:
        print(f"\ndisagreements ({len(disagreements)}), widest first:")
        for gap, a, b, row in sorted(disagreements, key=lambda d: -d[0]):
            print(
                f"  gap {gap}  first={a} second={b}  {row['query']!r} / "
                f"{row['meme_id']}  {row['ocr_text'][:50]!r}"
            )
    else:
        print("\nno disagreements")


if __name__ == "__main__":
    main()
