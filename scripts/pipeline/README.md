# Offline pipeline

The four scripts that build what the API serves. Run them in this order — each one reads
the previous one's output:

```bash
python scripts/pipeline/scan_dataset.py       # images        -> data/memes.jsonl
python scripts/pipeline/run_ocr_fast.py       # + OCR text    -> data/memes_ocr.jsonl
python scripts/pipeline/embed_memes_fast.py   # + CLIP vectors -> data/{text,image}.index, id_map.json
python scripts/pipeline/store_metadata.py     # metadata      -> MongoDB
```

You can skip all four: the FAISS indexes, `id_map.json` and `memes_final.jsonl` are
committed, so only `store_metadata.py` is needed to get search running. See the Quickstart
in the root README.

Each script is idempotent and safe to re-run.

## What is and isn't committed

`memes.jsonl` and `memes_ocr.jsonl` are intermediates and are **not** in the repo — each is
a strict subset of the committed `memes_final.jsonl`, so carrying them would be storing the
same OCR text three times. Running the pipeline from scratch recreates them, and running a
later step on its own tells you which earlier one to run first.

That matters most for `run_ocr_fast.py`: it is the expensive step, and its output is
preserved inside `memes_final.jsonl` under `text.ocr`. You do not need to redo the OCR pass
to get the captions back.
