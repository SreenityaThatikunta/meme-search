"""Measure the online search path: CLIP text encoding and FAISS retrieval.

Reports p50/p95 for each stage so the README's latency claim is reproducible.
MongoDB hydration is deliberately excluded — this measures retrieval, not full
request time.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python docs/benchmark.py

OMP_NUM_THREADS=1 is not a trick to flatter the numbers: a single short text
sequence does not parallelise, so the default thread pool measurably loses to one
thread (~27 ms vs ~19 ms p50) on contention alone.
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import clip
import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402 - needs the project root on sys.path first

QUERIES = [
    "exam stress",
    "monday morning",
    "one does not simply",
    "programming meme",
    "surprised cat",
    "relatable",
    "drink water",
    "distracted boyfriend",
    "this is fine",
    "spiderman pointing",
]
WARMUP = 5
ITERATIONS = 50
TOP_K = 5


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(int(len(ordered) * fraction) - 1, 0)]


def report(name: str, samples: list[float]) -> None:
    print(
        f"  {name:<34} p50 {statistics.median(samples):7.2f} ms"
        f"   p95 {percentile(samples, 0.95):7.2f} ms"
        f"   min {min(samples):7.2f} ms"
    )


def main() -> None:
    print(f"machine: {platform.machine()} | {platform.system()} {platform.release()}")
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()}")

    started = time.perf_counter()
    model, _ = clip.load(config.CLIP_MODEL, device=config.DEVICE)
    model.eval()
    print(f"model load: {time.perf_counter() - started:.2f}s")

    text_index = faiss.read_index(str(config.TEXT_INDEX_PATH))
    image_index = faiss.read_index(str(config.IMAGE_INDEX_PATH))
    with open(config.ID_MAP_PATH) as handle:
        id_map = json.load(handle)

    print(f"text.index:  {text_index.ntotal} vectors, dim {text_index.d}")
    print(f"image.index: {image_index.ntotal} vectors, dim {image_index.d}")
    print(f"id_map:      {len(id_map)} entries")

    def encode(query: str):
        tokens = clip.tokenize([query], truncate=True).to(config.DEVICE)
        with torch.no_grad():
            embedding = model.encode_text(tokens).cpu().numpy().astype("float32")
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding

    oversampled = TOP_K * config.HYBRID_OVERSAMPLE
    encode_ms, text_ms, image_ms, hybrid_ms = [], [], [], []

    for iteration in range(WARMUP + ITERATIONS):
        query = QUERIES[iteration % len(QUERIES)]

        start = time.perf_counter()
        embedding = encode(query)
        elapsed_encode = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        text_index.search(embedding, TOP_K)
        elapsed_text = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        image_index.search(embedding, oversampled)
        elapsed_image = (time.perf_counter() - start) * 1000

        # Full hybrid path: one encode, both indexes searched at oversampled depth.
        start = time.perf_counter()
        hybrid_embedding = encode(query)
        text_index.search(hybrid_embedding, oversampled)
        image_index.search(hybrid_embedding, oversampled)
        elapsed_hybrid = (time.perf_counter() - start) * 1000

        if iteration >= WARMUP:
            encode_ms.append(elapsed_encode)
            text_ms.append(elapsed_text)
            image_ms.append(elapsed_image)
            hybrid_ms.append(elapsed_hybrid)

    print(
        f"\n--- {ITERATIONS} iterations, {len(QUERIES)} distinct queries, "
        f"top_k={TOP_K} (hybrid oversamples to {oversampled}) ---"
    )
    report("CLIP text encode", encode_ms)
    report(f"FAISS text search (k={TOP_K})", text_ms)
    report(f"FAISS image search (k={oversampled})", image_ms)
    report("hybrid: encode + both searches", hybrid_ms)
    print("\nNOTE: excludes MongoDB hydration.")


if __name__ == "__main__":
    main()
