"""Evaluate retrieval quality per mode, and sweep the hybrid blend weight.

Reuses the production ranker (:func:`services.fusion.fuse_scores`) and the
production metrics module, so what is measured here is what the API serves.

Each query is encoded once and both indexes are searched once at oversampled
depth; every mode and every blend weight is then scored from those same
candidate lists. The sweep is therefore nearly free.

READ BEFORE TUNING: the eval set is built from each meme's own caption (see
scripts/eval/build_eval_set.py), so it structurally favours mode="text". A blend
weight chosen to maximise a score on this set would be overfitted to that
construction, not to real queries. The sweep is published to expose that bias,
not to pick a default.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/eval/evaluate_retrieval.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import clip
import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402 - needs the project root on sys.path first
from services.fusion import dedupe_candidates, fuse_ranks, fuse_scores  # noqa: E402
from services.metrics import summarise  # noqa: E402

EVAL_PATH = config.DATA_DIR / "eval_queries.jsonl"
KS = (1, 5, 10)
SWEEP = [round(w * 0.1, 1) for w in range(11)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 = all queries")
    parser.add_argument("--out", type=str, default="", help="write results as JSON")
    args = parser.parse_args()

    if not EVAL_PATH.exists():
        raise SystemExit(f"{EVAL_PATH} not found. Run scripts/eval/build_eval_set.py first.")

    with open(EVAL_PATH) as handle:
        queries = [json.loads(line) for line in handle]
    if args.limit:
        queries = queries[: args.limit]

    model, _ = clip.load(config.CLIP_MODEL, device=config.DEVICE)
    model.eval()
    text_index = faiss.read_index(str(config.TEXT_INDEX_PATH))
    image_index = faiss.read_index(str(config.IMAGE_INDEX_PATH))
    with open(config.ID_MAP_PATH) as handle:
        id_map = json.load(handle)

    depth = max(KS) * config.HYBRID_OVERSAMPLE
    print(f"queries: {len(queries)} | retrieval depth per index: {depth}")
    print(f"indexes: text {text_index.ntotal} | image {image_index.ntotal}")

    def resolve(scores, ids) -> list[tuple[str, float]]:
        hits = []
        for vector_id, score in zip(ids[0], scores[0], strict=False):
            if vector_id < 0:
                continue
            meme_id = id_map.get(str(int(vector_id)))
            if meme_id is not None:
                hits.append((meme_id, float(score)))
        return hits

    text_hits, image_hits, gold = [], [], []
    started = time.perf_counter()
    for position, row in enumerate(queries, start=1):
        tokens = clip.tokenize([row["query"]], truncate=True).to(config.DEVICE)
        with torch.no_grad():
            embedding = model.encode_text(tokens).cpu().numpy().astype("float32")
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)

        text_hits.append(resolve(*text_index.search(embedding, depth)))
        image_hits.append(resolve(*image_index.search(embedding, depth)))
        gold.append(row["gold_meme_id"])

        if position % 200 == 0:
            print(f"  encoded {position}/{len(queries)}")
    print(f"encode + search: {time.perf_counter() - started:.1f}s")

    def rank(hits: list[tuple[str, float]]) -> list[str]:
        return [meme_id for meme_id, _ in dedupe_candidates(hits)]

    results = {}
    results["text"] = summarise([rank(h) for h in text_hits], gold, KS)
    results["image"] = summarise([rank(h) for h in image_hits], gold, KS)

    for label, fuse in (("sum", fuse_scores), ("rrf", fuse_ranks)):
        for weight in SWEEP:
            rankings = [
                [meme_id for meme_id, _ in fuse(t, i, text_weight=weight)]
                for t, i in zip(text_hits, image_hits, strict=True)
            ]
            results[f"hybrid-{label}@{weight}"] = summarise(rankings, gold, KS)

    header = f"{'mode':<16}" + "".join(f"{'R@' + str(k):>10}" for k in KS) + f"{'MRR':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for name in ("text", "image"):
        row = results[name]
        line = f"{name:<16}" + "".join(f"{row['recall@' + str(k)]:>10.3f}" for k in KS)
        print(line + f"{row['mrr']:>10.3f}")
    for label, title in (("sum", "weighted score sum"), ("rrf", "reciprocal rank fusion")):
        print("-" * len(header) + f"  {title}")
        best = max(SWEEP, key=lambda w: results[f"hybrid-{label}@{w}"]["mrr"])
        for weight in SWEEP:
            row = results[f"hybrid-{label}@{weight}"]
            name = f"{label} w={weight}"
            line = f"{name:<16}" + "".join(f"{row['recall@' + str(k)]:>10.3f}" for k in KS)
            mark = "   <- best MRR" if weight == best else ""
            print(line + f"{row['mrr']:>10.3f}{mark}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
