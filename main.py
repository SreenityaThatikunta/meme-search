"""FastAPI service exposing semantic meme search over HTTP."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from services import search as search_service

# CLIP/FAISS both link against OpenMP; on macOS the duplicate runtime check
# aborts the process without this escape hatch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=os.getenv("MEME_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and indexes once, before the first request is served."""
    try:
        search_service.load_resources()
        logger.info("Resources loaded, server ready")
    except Exception:
        logger.exception("Failed to load search resources")
        raise
    yield


app = FastAPI(
    title="Meme Search API",
    description=(
        "Semantic meme search over CLIP embeddings and FAISS indexes, with "
        "OCR-text, image and hybrid retrieval modes."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# The Chrome extension and the Streamlit UI are served from other origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("MEME_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve meme images so clients can render results without filesystem access.
if config.IMAGE_DIR.is_dir():
    app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")
else:
    logger.warning(
        "Image directory %s not found; /images will not be served. "
        "Set MEME_IMAGE_DIR or download the dataset.",
        config.IMAGE_DIR,
    )


class SearchRequest(BaseModel):
    """Body of a ``POST /search/text`` request."""

    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(
        config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K, description="Results to return"
    )
    mode: Literal["text", "image", "hybrid"] = Field(
        "hybrid", description="Retrieval mode: OCR text, image, or both fused"
    )
    text_weight: float = Field(
        config.HYBRID_TEXT_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight for the OCR-text index when mode='hybrid'",
    )


class SearchResult(BaseModel):
    """One ranked meme."""

    meme_id: str
    image_path: str
    image_url: str
    ocr_text: str = ""
    score: float


@app.get("/health")
def health() -> dict:
    """Report which resources loaded and which retrieval modes are available."""
    return search_service.health()


@app.post("/search/text", response_model=list[SearchResult])
def text_search(payload: SearchRequest) -> list[dict]:
    """Rank memes against a natural language query."""
    try:
        return search_service.search(
            payload.query,
            top_k=payload.top_k,
            mode=payload.mode,
            text_weight=payload.text_weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search failed for query %r", payload.query)
        raise HTTPException(status_code=500, detail="Internal search error") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT)
