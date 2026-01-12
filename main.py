from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.search import search_text, load_resources

app = FastAPI()


# -----------------------------
# REQUEST SCHEMA (IMPORTANT)
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