import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.search import search_text, load_resources

# -----------------------------
# ENV FIX (OpenMP safety on macOS)
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()


# -----------------------------
# CORS (REQUIRED FOR EXTENSION)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # OK for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# STATIC IMAGE SERVING
# -----------------------------
# This exposes images at:
# http://127.0.0.1:8000/images/...
app.mount(
    "/images",
    StaticFiles(directory="data/images/images"),
    name="images"
)


# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup_event():
    try:
        load_resources()
        print("[INFO] Resources loaded, server ready")
    except Exception as e:
        print("[FATAL] Failed to load resources:", e)
        raise


# -----------------------------
# ENDPOINT
# -----------------------------
@app.post("/search/text")
def text_search(payload: TextSearchRequest):
    try:
        return search_text(payload.query, payload.top_k)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        print("[ERROR] search_text crashed:", e)
        raise HTTPException(status_code=500, detail=str(e))