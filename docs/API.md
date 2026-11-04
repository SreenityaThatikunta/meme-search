# API reference

Two endpoints. FastAPI generates interactive docs from the same source at
**`http://localhost:8000/docs`** — that's authoritative and always current; this page is
for reading without starting the server.

Defined in [`main.py`](../main.py).

## `POST /search/text`

Rank memes against a natural language query.

```bash
curl -s localhost:8000/search/text -H 'content-type: application/json' \
  -d '{"query": "exam stress", "top_k": 5, "mode": "hybrid"}'
```

### Request

| Field | Type | Default | Constraints |
|---|---|---|---|
| `query` | string | required | At least 1 character |
| `top_k` | int | `5` | 1–50 (`MEME_MAX_TOP_K`) |
| `mode` | enum | `"hybrid"` | `text`, `image` or `hybrid` |
| `text_weight` | float | `0.2` | 0.0–1.0; hybrid only |

`text_weight` is the weight given to the OCR-caption index — `0.0` ranks on appearance
alone, `1.0` on caption alone. The `0.2` default is measured rather than guessed; see
[EVALUATION.md](EVALUATION.md).

### Response

`200` with a JSON array, ranked best first:

```json
[
  {
    "meme_id": "meme_000042",
    "image_path": "data/images/images/image_107.jpg",
    "image_url": "http://127.0.0.1:8000/images/image_107.jpg",
    "ocr_text": "me studying the night before the exam",
    "score": 0.2841
  }
]
```

`score` is a cosine similarity in `text`/`image` mode, or a fused rank score in `hybrid`
mode. Either way it's comparable **within** one result set and is not a calibrated
probability — don't threshold on it across queries.

An empty array means nothing matched (or the query was whitespace). That's a `200`, not a
`404`.

### Errors

| Status | When |
|---|---|
| `400` | Unknown `mode`, or a mode whose index isn't loaded |
| `422` | Request failed schema validation (`top_k` out of range, missing `query`) |
| `503` | A required index file is missing — the service is misconfigured, not the request |
| `500` | Anything unexpected. Detail is logged server-side, not returned to the caller |

## `GET /health`

Readiness snapshot. Both clients call this on load to disable modes whose index is missing,
rather than failing at query time.

```bash
curl -s localhost:8000/health
```

```json
{
  "model_loaded": true,
  "clip_model": "ViT-B/32",
  "device": "cpu",
  "text_index_vectors": 6989,
  "image_index_vectors": 6989,
  "id_map_entries": 6989,
  "mongo_connected": true,
  "mongo_error": null,
  "available_modes": ["text", "image", "hybrid"]
}
```

`available_modes` is the useful field: if the image index failed to load it reports
`["text"]` only, and `mongo_connected: false` means searches will run but return nothing,
because results can't be hydrated with paths and captions.

## `GET /images/{filename}`

Static mount over `MEME_IMAGE_DIR`, so clients can render results without filesystem
access to the dataset. Skipped with a warning if the image directory doesn't exist — the
API still serves search, and `image_url`s will 404.
