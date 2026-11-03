"""Render one labelled contact sheet per relevance query.

Judging 156 pooled pairs one image at a time is slow and invites drift in the
judge's standard. A contact sheet puts every candidate for one query side by
side, so each query is judged in a single consistent pass, and the numbering
lines up with the index printed by scripts/show_pool.py.

Usage:
    python scripts/build_contact_sheets.py --out <dir>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402 - needs the project root on sys.path first

ImageFile.LOAD_TRUNCATED_IMAGES = True

POOL_PATH = config.DATA_DIR / "relevance_pool.jsonl"
COLUMNS = 4
CELL = 360
LABEL_H = 28
PAD = 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pool",
        default=str(POOL_PATH),
        help="pool file to render (default: the full relevance pool)",
    )
    args = parser.parse_args()
    pool_path = Path(args.pool)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pool_path.exists():
        raise SystemExit(f"{pool_path} not found. Run build_relevance_pool.py first.")

    by_query: dict[str, list[dict]] = defaultdict(list)
    with open(pool_path) as handle:
        for line in handle:
            row = json.loads(line)
            by_query[row["query"]].append(row)

    for query, rows in by_query.items():
        count = len(rows)
        cols = min(COLUMNS, count)
        rows_n = (count + cols - 1) // cols
        width = cols * (CELL + PAD) + PAD
        height = rows_n * (CELL + LABEL_H + PAD) + PAD

        sheet = Image.new("RGB", (width, height), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)

        for index, row in enumerate(rows):
            col, line_no = index % cols, index // cols
            x = PAD + col * (CELL + PAD)
            y = PAD + line_no * (CELL + LABEL_H + PAD)

            draw.rectangle([x, y, x + CELL, y + LABEL_H], fill=(30, 30, 30))
            label = row.get("item", index + 1)
            draw.text((x + 8, y + 7), f"[{label}]  {row['meme_id']}", fill=(255, 255, 255))

            try:
                with Image.open(row["image_path"]) as img:
                    thumb = img.convert("RGB")
                    thumb.thumbnail((CELL, CELL))
                    offset_x = x + (CELL - thumb.width) // 2
                    offset_y = y + LABEL_H + (CELL - thumb.height) // 2
                    sheet.paste(thumb, (offset_x, offset_y))
            except Exception as exc:
                draw.text((x + 8, y + LABEL_H + 8), f"unreadable: {exc}", fill=(180, 0, 0))

        safe = query.replace(" ", "_").replace("/", "_")
        path = out_dir / f"{safe}.png"
        sheet.save(path, optimize=True)
        print(f"  {path.name}  ({count} candidates)")

    print(f"\n{len(by_query)} sheets written to {out_dir}")


if __name__ == "__main__":
    main()
