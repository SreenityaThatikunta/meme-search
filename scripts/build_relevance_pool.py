"""Pool candidates for open-ended relevance judging, blind.

Known-item evaluation (scripts/build_eval_set.py) has one correct answer per
query. Open-ended queries do not: many memes can legitimately answer "monday
morning". Measuring those needs graded human judgments, which needs a candidate
pool to judge.

Standard TREC-style pooling is used: take the top-k from every system under
comparison, union them, and judge the union. A meme no system retrieved is
assumed irrelevant — the usual, and documented, pooling assumption.

Candidates are **shuffled with a fixed seed** before being written out, and the
file records no system attribution. The judge therefore cannot tell which mode
produced a candidate, so judgments cannot flatter a favoured system. The
system-to-candidate mapping is kept separately for scoring.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/build_relevance_pool.py
"""

import argparse
import json
import random
import sys
from pathlib import Path

import clip
import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402 - needs the project root on sys.path first
from services.fusion import dedupe_candidates, fuse_ranks  # noqa: E402

QUERY_PATH = config.DATA_DIR / "relevance_queries.txt"
POOL_PATH = config.DATA_DIR / "relevance_pool.jsonl"
SYSTEMS_PATH = config.DATA_DIR / "relevance_systems.json"
POOL_DEPTH = 5
SEED = 7


def load_queries() -> list[str]:
    if not QUERY_PATH.exists():
        raise SystemExit(f"{QUERY_PATH} not found.")
    lines = []
    for raw in QUERY_PATH.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    args = parser.parse_args()

    queries = load_queries()
    model, _ = clip.load(config.CLIP_MODEL, device=config.DEVICE)
    model.eval()
    text_index = faiss.read_index(str(config.TEXT_INDEX_PATH))
    image_index = faiss.read_index(str(config.IMAGE_INDEX_PATH))
    with open(config.ID_MAP_PATH) as handle:
        id_map = json.load(handle)
    with open(config.METADATA_PATH) as handle:
        meta = {record["meme_id"]: record for record in (json.loads(line) for line in handle)}

    rng = random.Random(SEED)
    fetch_k = args.depth * config.HYBRID_OVERSAMPLE

    def resolve(scores, ids) -> list[tuple[str, float]]:
        hits = []
        for vector_id, score in zip(ids[0], scores[0], strict=False):
            if vector_id < 0:
                continue
            meme_id = id_map.get(str(int(vector_id)))
            if meme_id is not None:
                hits.append((meme_id, float(score)))
        return hits

    pool_rows, systems = [], {}

    for query in queries:
        tokens = clip.tokenize([query], truncate=True).to(config.DEVICE)
        with torch.no_grad():
            embedding = model.encode_text(tokens).cpu().numpy().astype("float32")
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)

        text_hits = resolve(*text_index.search(embedding, fetch_k))
        image_hits = resolve(*image_index.search(embedding, fetch_k))

        ranked = {
            "text": [m for m, _ in dedupe_candidates(text_hits)][: args.depth],
            "image": [m for m, _ in dedupe_candidates(image_hits)][: args.depth],
        }
        for weight in (0.1, 0.2, 0.5):
            fused = fuse_ranks(text_hits, image_hits, weight, config.RRF_K)
            ranked[f"rrf@{weight}"] = [m for m, _ in fused][: args.depth]

        systems[query] = ranked

        pooled = sorted({meme_id for ids in ranked.values() for meme_id in ids})
        rng.shuffle(pooled)  # blind the judge to which system found what

        for meme_id in pooled:
            record = meta.get(meme_id, {})
            pool_rows.append(
                {
                    "query": query,
                    "meme_id": meme_id,
                    "image_path": record.get("image_path", ""),
                    "ocr_text": record.get("text", {}).get("ocr", ""),
                    "relevance": None,  # to be filled by a judge: 0, 1 or 2
                }
            )

    with open(POOL_PATH, "w") as handle:
        for row in pool_rows:
            handle.write(json.dumps(row) + "\n")
    SYSTEMS_PATH.write_text(json.dumps(systems, indent=2))

    per_query = len(pool_rows) / len(queries)
    print(f"queries: {len(queries)} | systems per query: {len(systems[queries[0]])}")
    print(f"pooled pairs to judge: {len(pool_rows)} ({per_query:.1f} per query)")
    print(f"wrote {POOL_PATH} (shuffled, no system attribution)")
    print(f"wrote {SYSTEMS_PATH} (rankings, for scoring)")


if __name__ == "__main__":
    main()
