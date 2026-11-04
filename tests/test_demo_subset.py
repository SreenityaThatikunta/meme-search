"""Tests for the demo subset builder.

The invariant under test is the one from docs/ARCHITECTURE.md: row *i* of
text.index and row *i* of image.index must describe the same meme, and
id_map.json must resolve row *i* to that meme. Getting this wrong does not
raise — it produces a demo that returns confident, wrong results — so it is
worth a test rather than a careful read.

faiss and numpy are not in requirements-dev.txt (CI runs the suite without the
heavy ML stack), so these skip in CI and run locally wherever faiss is present.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

faiss = pytest.importorskip("faiss", reason="faiss-cpu is not a dev dependency")
np = pytest.importorskip("numpy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    """Import the builder by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "build_demo_subset", PROJECT_ROOT / "scripts" / "demo" / "build_demo_subset.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_demo_subset"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def full_dataset(tmp_path):
    """A miniature stand-in for the real indexes and metadata."""
    dim = 8
    count = 10
    rng = np.random.default_rng(0)

    text_vectors = rng.normal(size=(count, dim)).astype("float32")
    image_vectors = rng.normal(size=(count, dim)).astype("float32")
    for vectors in (text_vectors, image_vectors):
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    for name, vectors in (("text.index", text_vectors), ("image.index", image_vectors)):
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        faiss.write_index(index, str(tmp_path / name))

    records = [
        {
            "meme_id": f"meme_{i:06d}",
            "image_path": f"data/images/images/image_{i}.jpg",
            "text": {"ocr": f"caption number {i} with enough words"},
            "vector_id": i,
        }
        for i in range(count)
    ]
    with open(tmp_path / "memes_final.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    return {
        "dir": tmp_path,
        "dim": dim,
        "count": count,
        "text_vectors": text_vectors,
        "image_vectors": image_vectors,
        "records": records,
    }


def test_slice_index_preserves_vectors_in_the_requested_order(full_dataset):
    builder = load_builder()
    wanted = [7, 2, 5]

    sliced = builder.slice_index(full_dataset["dir"] / "text.index", wanted, full_dataset["dim"])

    assert sliced.ntotal == 3
    for row, original_id in enumerate(wanted):
        assert np.allclose(
            sliced.reconstruct(row), full_dataset["text_vectors"][original_id], atol=1e-6
        )


def test_slice_index_rejects_a_dimension_mismatch(full_dataset):
    builder = load_builder()
    with pytest.raises(SystemExit, match="expected"):
        builder.slice_index(full_dataset["dir"] / "text.index", [0], 512)


def test_both_indexes_and_the_id_map_stay_aligned(full_dataset, monkeypatch, tmp_path):
    """The load-bearing invariant: row i is the same meme everywhere."""
    builder = load_builder()
    data = full_dataset

    monkeypatch.setattr(builder.config, "DATA_DIR", data["dir"])
    monkeypatch.setattr(builder.config, "METADATA_PATH", data["dir"] / "memes_final.jsonl")
    monkeypatch.setattr(builder.config, "TEXT_INDEX_PATH", data["dir"] / "text.index")
    monkeypatch.setattr(builder.config, "IMAGE_INDEX_PATH", data["dir"] / "image.index")
    monkeypatch.setattr(builder.config, "IMAGE_DIR", data["dir"] / "images")
    monkeypatch.setattr(builder.config, "EMBEDDING_DIM", data["dim"])
    monkeypatch.setattr(builder.config, "PROJECT_ROOT", tmp_path)

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["build_demo_subset.py", "--size", "4", "--out", str(out)])
    builder.main()

    text_index = faiss.read_index(str(out / "text.index"))
    image_index = faiss.read_index(str(out / "image.index"))
    id_map = json.loads((out / "id_map.json").read_text())
    subset = [json.loads(line) for line in open(out / "memes_final.jsonl")]

    # Every artefact agrees on how many memes there are.
    assert text_index.ntotal == image_index.ntotal == len(id_map) == len(subset) == 4

    originals = {record["meme_id"]: record["vector_id"] for record in data["records"]}
    for row in range(text_index.ntotal):
        meme_id = id_map[str(row)]
        original_id = originals[meme_id]

        # Row i of BOTH indexes holds that meme's original vectors.
        assert np.allclose(
            text_index.reconstruct(row), data["text_vectors"][original_id], atol=1e-6
        )
        assert np.allclose(
            image_index.reconstruct(row), data["image_vectors"][original_id], atol=1e-6
        )

    # vector_id is renumbered to the new row, not left pointing at the old one.
    assert [record["vector_id"] for record in subset] == [0, 1, 2, 3]
    assert [id_map[str(i)] for i in range(4)] == [record["meme_id"] for record in subset]


def test_selection_is_deterministic_for_a_fixed_seed(full_dataset, monkeypatch):
    builder = load_builder()
    monkeypatch.setattr(builder.config, "DATA_DIR", full_dataset["dir"])
    records = {record["meme_id"]: record for record in full_dataset["records"]}

    first = builder.select_memes(records, size=5, seed=42)
    second = builder.select_memes(records, size=5, seed=42)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5  # no duplicates between the pool and the top-up
