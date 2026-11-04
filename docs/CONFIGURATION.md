# Configuration

Everything is environment-driven with local defaults, so nothing needs editing to run
locally and nothing needs editing to containerise. [`config.py`](../config.py) is the single
source of truth; [`.env.example`](../.env.example) lists every setting with its default.

The app reads **real environment variables, not the `.env` file** — `export` what you need,
or pass them via `docker compose`.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `MEME_DATA_DIR` | `data` | Root for data artefacts |
| `MEME_IMAGE_DIR` | `data/images/images` | Images served at `/images` |
| `MEME_TEXT_INDEX` | `data/text.index` | FAISS index of OCR-caption embeddings |
| `MEME_IMAGE_INDEX` | `data/image.index` | FAISS index of image embeddings |
| `MEME_ID_MAP` | `data/id_map.json` | Vector row number → meme id |
| `MEME_METADATA` | `data/memes_final.jsonl` | Source for the MongoDB load |
| `MEME_MONGO_URI` | `mongodb://localhost:27017` | Metadata store |
| `MEME_MONGO_DB` | `meme_search` | Database name |
| `MEME_MONGO_COLLECTION` | `memes` | Collection name |
| `MEME_CLIP_MODEL` | `ViT-B/32` | CLIP variant; must match what built the indexes |
| `MEME_DEVICE` | `cpu` | `cpu`, `mps` or `cuda` |
| `MEME_API_HOST` | `0.0.0.0` | Bind address |
| `MEME_API_PORT` | `8000` | Bind port |
| `MEME_PUBLIC_BASE_URL` | `http://127.0.0.1:8000` | Base for returned `image_url`s |
| `MEME_DEFAULT_TOP_K` | `5` | Results when the request omits `top_k` |
| `MEME_MAX_TOP_K` | `50` | Hard cap on `top_k` |
| `MEME_HYBRID_TEXT_WEIGHT` | `0.2` | Default hybrid blend — measured, see [EVALUATION.md](EVALUATION.md) |
| `MEME_FUSION_STRATEGY` | `rrf` | `rrf` (rank fusion) or `sum` (weighted cosines) |
| `MEME_RRF_K` | `60` | RRF damping constant |
| `MEME_HYBRID_OVERSAMPLE` | `4` | Candidates pulled per index before fusing, as a multiple of `top_k` |
| `MEME_CORS_ORIGINS` | `*` | Comma-separated allowlist. **Tighten before exposing the API** |
| `MEME_LOG_LEVEL` | `INFO` | Standard Python log levels |
| `MEME_API_BASE_URL` | `http://127.0.0.1:8000` | Read by the Streamlit UI to find the API |

## Two settings worth understanding

**`MEME_FUSION_STRATEGY`** changes how hybrid results are combined. `rrf` fuses ranks and
is the default because it's immune to CLIP's modality gap; `sum` fuses raw cosines and is
kept so the evaluation harness can keep measuring the alternative it replaced. Switching to
`sum` costs about 20 points of Recall@10 — see
[OVERVIEW.md](OVERVIEW.md#step-4-combine-the-two-rankings--the-part-that-was-wrong-at-first).

**`MEME_CLIP_MODEL`** must match the model that built the committed indexes (`ViT-B/32`).
Changing it without re-running `scripts/pipeline/embed_memes_fast.py` compares query
vectors against embeddings from a different model, and results become noise rather than
erroring.

## A note on installation

`requirements.txt` installs OpenAI's CLIP from git, and its `setup.py` still imports
`pkg_resources`, which setuptools removed in v81. On a fresh Python 3.12+ environment a
plain `pip install -r requirements.txt` therefore fails with
`ModuleNotFoundError: No module named 'pkg_resources'`. Pinning setuptools below 81 and
disabling build isolation makes the install reproducible:

```bash
pip install "setuptools<81" wheel
pip install --no-build-isolation -r requirements.txt
```
