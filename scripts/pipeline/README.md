# Offline pipeline

The four scripts that build what the API serves. Run them in this order — each one reads
the previous one's output:

```bash
python scripts/pipeline/scan_dataset.py       # images        -> data/memes.jsonl
python scripts/pipeline/run_ocr_fast.py       # + OCR text    -> data/memes_ocr.jsonl
python scripts/pipeline/embed_memes_fast.py   # + CLIP vectors -> data/{text,image}.index, id_map.json
python scripts/pipeline/store_metadata.py     # metadata      -> MongoDB
```

You can skip all four: the FAISS indexes and `id_map.json` are committed, so only
`store_metadata.py` is needed to get search running. See the Quickstart in the root README.

Each script is idempotent and safe to re-run.
