# Deploying the demo to Hugging Face Spaces

The Space is a separate git repo from this one. This repo holds the code and the script
that builds the demo payload; the Space holds a copy of the code plus the 400-meme subset.

`demo/data/` is gitignored here on purpose — committing 40 MB of images to GitHub would
recreate the data-dump problem, and it is regenerable in under a minute.

## 1. Build the payload

```bash
python scripts/demo/build_demo_subset.py
```

Writes `demo/data/` — two sliced FAISS indexes (~820 KB each, down from 14 MB), an
`id_map.json`, a 400-record `memes_final.jsonl`, and the 400 image files. About 41 MB.

Needs `faiss-cpu` and `numpy`, plus `data/images/` present locally. It does **not** need
CLIP or a GPU: `IndexFlatIP` stores vectors verbatim, so the subset is sliced out of the
committed indexes rather than re-embedded.

## 2. Check it locally first

```bash
streamlit run demo/space_app.py
```

This is the same entry point the Space uses, so if it works here it works there. Search
`exam stress`, switch all three modes, confirm images render.

## 3. Create the Space

On huggingface.co: **New Space** → SDK **Streamlit** → hardware **CPU basic (free)**. Call
it `meme-search`. You need an HF account; nothing here can create one for you.

## 4. Assemble and push

```bash
git clone https://huggingface.co/spaces/<your-username>/meme-search /tmp/meme-space
```

Copy in the code, the Space's own README and requirements, and the payload at the default
`data/` path — which is why no environment variables need setting:

```bash
cd /Users/sreenityathatikunta/Documents/Projects/meme-search && cp app.py config.py /tmp/meme-space/ && cp -r services /tmp/meme-space/ && mkdir -p /tmp/meme-space/demo && cp demo/space_app.py /tmp/meme-space/demo/ && cp demo/README.md demo/requirements.txt /tmp/meme-space/ && cp -r demo/data/. /tmp/meme-space/data/
```

The two 820 KB indexes are under HF's 10 MB LFS threshold, so no LFS setup is needed:

```bash
cd /tmp/meme-space && git add -A && git commit -m "Deploy meme search demo" && git push
```

Watch the build log on the Space page. First build takes a few minutes — most of it is
installing torch.

## 5. Wire it up

Once it's live:

- Replace the "No hosted demo" note at the top of the root `README.md` with the Space link
- Repoint the repo's website: `gh repo edit --homepage "https://huggingface.co/spaces/<your-username>/meme-search"`
- Take the screenshot from the Space and drop it at `docs/images/demo.png` — item 1 of the
  recruiter checklist closes with it

## If the build fails

**`sdk_version` not supported** — the most likely failure, and the only field I couldn't
verify against HF's current list. Check the supported Streamlit versions on the Space's
settings page and edit the frontmatter in `README.md` to match.

**Torch pulling CUDA wheels / out of disk** — confirm the `--extra-index-url` line is the
first line of `requirements.txt`.

**Space builds but every result scores oddly** — the `clip` pin drifted from the root
`requirements.txt`. The committed vectors came from that exact commit; a different CLIP
build produces a silently incompatible query embedding.
