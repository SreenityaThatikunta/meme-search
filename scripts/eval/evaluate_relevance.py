"""Score the open-ended relevance set: nDCG@5 and P@5 per mode.

Complements scripts/eval/evaluate_retrieval.py. That one measures known-item
retrieval (one correct answer per query); this one measures graded relevance on
open-ended queries, where many memes can legitimately answer.

Reads the judgments written into data/relevance_pool.jsonl and the system
rankings recorded in data/relevance_systems.json, so it needs no model and no
indexes — it runs anywhere, in a second.

Usage:
    python scripts/eval/evaluate_relevance.py [--out results.json]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402 - needs the project root on sys.path first
from services.metrics import ndcg_at_k, precision_at_k  # noqa: E402

POOL_PATH = config.DATA_DIR / "relevance_pool.jsonl"
SYSTEMS_PATH = config.DATA_DIR / "relevance_systems.json"
K = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    for path in (POOL_PATH, SYSTEMS_PATH):
        if not path.exists():
            raise SystemExit(f"{path} not found. Run build_relevance_pool.py first.")

    judgments: dict[str, dict[str, int]] = defaultdict(dict)
    unjudged = 0
    with open(POOL_PATH) as handle:
        for line in handle:
            row = json.loads(line)
            if row["relevance"] is None:
                unjudged += 1
                continue
            judgments[row["query"]][row["meme_id"]] = int(row["relevance"])

    if unjudged:
        raise SystemExit(f"{unjudged} pooled pairs are unjudged; fill in 'relevance'.")

    systems = json.loads(SYSTEMS_PATH.read_text())
    queries = sorted(systems)
    names = sorted(next(iter(systems.values())))

    results, per_query = {}, {}
    for name in names:
        ndcgs, precisions = [], []
        for query in queries:
            ranking = systems[query][name]
            judged = judgments[query]
            ndcgs.append(ndcg_at_k(ranking, judged, K))
            precisions.append(precision_at_k(ranking, judged, K))
        results[name] = {
            f"ndcg@{K}": statistics.mean(ndcgs),
            f"p@{K}": statistics.mean(precisions),
            "queries": len(queries),
        }
        per_query[name] = dict(zip(queries, ndcgs, strict=True))

    header = f"{'mode':<16}{'nDCG@5':>10}{'P@5':>10}"
    print(f"{header}\n{'-' * len(header)}")
    best = max(names, key=lambda n: results[n][f"ndcg@{K}"])
    for name in names:
        row = results[name]
        mark = "   <- best" if name == best else ""
        print(f"{name:<16}{row[f'ndcg@{K}']:>10.3f}{row[f'p@{K}']:>10.3f}{mark}")

    # Per-query nDCG for the default config exposes which query types fail.
    default = f"rrf@{config.HYBRID_TEXT_WEIGHT}"
    if default in per_query:
        print(f"\nper-query nDCG@5 for {default} (worst first):")
        for query, score in sorted(per_query[default].items(), key=lambda kv: kv[1]):
            print(f"  {score:.3f}  {query}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": results, "per_query": per_query}, indent=2)
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
