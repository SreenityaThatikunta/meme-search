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

# --- MongoDB ---
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
# Weight given to the OCR-text index when fusing hybrid results (0.0-1.0).
# The image index receives the remaining weight.
HYBRID_TEXT_WEIGHT = float(os.getenv("MEME_HYBRID_TEXT_WEIGHT", "0.5"))
# How many extra candidates to pull from each index before fusing.
HYBRID_OVERSAMPLE = int(os.getenv("MEME_HYBRID_OVERSAMPLE", "4"))
