"""Central configuration for the meme search engine.

Every path, host and model name is read from the environment with a sensible
local default, so the same code runs unchanged on a laptop, in Docker, or in CI.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _path(env_var: str, default: str) -> Path:
    """Resolve a path setting, relative to the project root when not absolute."""
    value = os.getenv(env_var)
    if value:
        return Path(value).expanduser()
    return PROJECT_ROOT / default


# --- Data and index locations ---
DATA_DIR = _path("MEME_DATA_DIR", "data")
IMAGE_DIR = _path("MEME_IMAGE_DIR", "data/images/images")
TEXT_INDEX_PATH = _path("MEME_TEXT_INDEX", "data/text.index")
IMAGE_INDEX_PATH = _path("MEME_IMAGE_INDEX", "data/image.index")
ID_MAP_PATH = _path("MEME_ID_MAP", "data/id_map.json")
METADATA_PATH = _path("MEME_METADATA", "data/memes_final.jsonl")

# --- Metadata store ---
# Where the search path reads image paths and captions from: "mongo" (default)
# or "jsonl", which reads METADATA_PATH into memory and needs no database. The
# jsonl backend suits read-only deployments where metadata never changes after
# the indexes are built; see services/metadata.py.
METADATA_BACKEND = os.getenv("MEME_METADATA_BACKEND", "mongo")

# --- MongoDB (used when METADATA_BACKEND == "mongo") ---
MONGO_URI = os.getenv("MEME_MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MEME_MONGO_DB", "meme_search")
MONGO_COLLECTION = os.getenv("MEME_MONGO_COLLECTION", "memes")

# --- Model ---
CLIP_MODEL = os.getenv("MEME_CLIP_MODEL", "ViT-B/32")
DEVICE = os.getenv("MEME_DEVICE", "cpu")
EMBEDDING_DIM = 512

# --- Serving ---
API_HOST = os.getenv("MEME_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("MEME_API_PORT", "8000"))
# Public base URL used to build image links returned by the API.
PUBLIC_BASE_URL = os.getenv("MEME_PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# --- Search behaviour ---
DEFAULT_TOP_K = int(os.getenv("MEME_DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MEME_MAX_TOP_K", "50"))
# How to combine the two indexes' candidate lists: "rrf" (reciprocal rank
# fusion, scale-free) or "sum" (weighted cosine sum). RRF is the default because
# text-to-text and text-to-image cosines differ ~3x in magnitude, which makes a
# weighted sum unstable; see docs/EVALUATION.md.
FUSION_STRATEGY = os.getenv("MEME_FUSION_STRATEGY", "rrf")
# RRF damping constant from the original paper.
RRF_K = int(os.getenv("MEME_RRF_K", "60"))
# Weight given to the OCR-text index when fusing hybrid results (0.0-1.0).
# The image index receives the remaining weight. 0.2 maximises Recall@5/@10 on
# the known-item eval set; see docs/EVALUATION.md.
HYBRID_TEXT_WEIGHT = float(os.getenv("MEME_HYBRID_TEXT_WEIGHT", "0.2"))
# How many extra candidates to pull from each index before fusing.
HYBRID_OVERSAMPLE = int(os.getenv("MEME_HYBRID_OVERSAMPLE", "4"))
