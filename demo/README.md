---
title: Meme Search
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.52.2
app_file: demo/space_app.py
pinned: false
license: mit
short_description: Search memes by meaning with CLIP embeddings and FAISS
---

# Meme Search — live demo

Search memes by describing them. Type `exam stress` or `monday morning`; no tags, no
filenames.

Each meme is indexed twice with CLIP — once as the image, once as the text OCR'd off it —
into two FAISS indexes, and the two rankings are combined with weighted Reciprocal Rank
Fusion. Switch **Mode** to see the difference:

- `text` matches the caption scraped off the meme
- `image` matches what the meme looks like, ignoring its words
- `hybrid` fuses both, and is the default

**This demo serves a 400-meme subset** so it fits a free CPU Space. The numbers reported
in the repo — Recall@10 0.599, p50 ~29 ms — were measured on the full 6,989-meme dataset.
Retrieval behaviour is identical; there is simply less to find, so expect thinner results
on narrow queries.

The first query after the Space wakes up includes loading CLIP, so give it a few seconds.
After that, queries are ~30 ms of search plus page render.

Source, architecture notes and the full evaluation:
**[github.com/SreenityaThatikunta/meme-search](https://github.com/SreenityaThatikunta/meme-search)**

Worth knowing: template queries are the weak spot. `this is fine` returns thematically
adjacent calamity memes rather than the burning-room dog — template identity is a
different retrieval problem from subject matter.
