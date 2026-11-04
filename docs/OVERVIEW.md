# How it works, in five minutes

Read this after the [README](../README.md) and before any code. It explains the whole
system in prose, assuming you know nothing about embeddings or vector search. When you're
done, [`services/search.py`](../services/search.py) and
[`services/fusion.py`](../services/fusion.py) will read as obvious.

## The problem

You have 7,000 memes in a folder. They're named `image_1.jpg` through `image_6992.jpg`.
You want the one about Monday mornings.

Filename search is useless. Full-text search is useless too — there's no text to search,
because the caption is *pixels*, burned into the image. And the joke isn't only in the
caption anyway. It's split across two places at once:

- **The picture.** The burning-room dog, the distracted boyfriend, the surprised cat. You
  might remember the template and not a single word of the text.
- **The caption.** `one does not simply` is funny because of the words; the Boromir still
  behind them is interchangeable.

Any system that indexes only one of those will reliably fail on the other half.

## Step 1: turn meaning into numbers

An **embedding** is a list of numbers that stands for the meaning of something. Feed in the
phrase "student burnout" and you get back 512 numbers. Feed in "me during finals week" and
you get back 512 different numbers — but *close* to the first set, because the meanings are
close. Feed in "tax policy" and you get numbers far away from both.

"Close" has a precise meaning here. Each embedding is a point in 512-dimensional space, and
closeness is **cosine similarity** — roughly, the angle between two points as seen from the
origin. Identical direction scores 1.0, unrelated scores near 0.

This is what makes search by meaning possible. You never compare words to words. You
convert the query into a point, convert everything you own into points ahead of time, and
return whatever sits nearest the query.

## Step 2: use a model that embeds pictures and words into the same space

Most embedding models handle one kind of input. A sentence model turns text into points; an
image model turns images into points; the two sets of points have nothing to do with each
other, because they were never trained to be comparable.

**CLIP** is different, and it's the reason this project works. CLIP was trained on hundreds
of millions of (image, caption) pairs with one objective: put an image and its real caption
close together, and push mismatched pairs apart. The result is a *single* 512-dimensional
space holding both kinds of thing.

The consequence is the trick at the centre of this repo: **you can compare a text query
directly against an image.** Type "surprised cat" and CLIP's text encoder gives you a point
that genuinely sits near the embeddings of cat-looking images — not near the embedding of
the *word* "cat", near the pictures. No uploaded reference image needed.

## Step 3: build two indexes, not one

So we embed each meme twice, into two separate indexes that share one numbering:

- **`text.index`** — the embedding of the OCR'd caption. We run EasyOCR over every image
  first to recover the words as text, then embed those words.
- **`image.index`** — the embedding of the raw pixels.

Each fails in a way the other survives:

| | Fails when | Example of the failure |
|---|---|---|
| `text.index` | OCR can't read it — stylised fonts, low contrast, text over busy artwork | A meme whose caption is in a hand-drawn font is invisible to it |
| `image.index` | The joke is purely linguistic | Two copies of the same template with opposite captions are near-identical vectors |

Searching both and combining them is strictly better than either, which is why `hybrid` is
the default mode.

Storing them is a **vector index** — FAISS — not a database. FAISS does exactly one thing:
hold vectors and return the nearest ones, as integer row numbers. It knows nothing about
file paths or captions. So row 4,213 comes back and something has to turn that into a meme.
That's `id_map.json` (row number → meme id) and MongoDB (meme id → path and caption). The
split is deliberate: metadata can be corrected without re-embedding 7,000 images, and
re-embedding doesn't disturb metadata.

## Step 4: combine the two rankings — the part that was wrong at first

Here's the most interesting bug in the project's history, and the thing worth understanding
if you understand nothing else.

Both indexes return cosine similarities. Both therefore produce scores in the range
`[-1, 1]`. It follows naturally that you can blend them with a weighted average:

```
score = w · cos(query, caption) + (1 − w) · cos(query, image)
```

**This is wrong,** and measuring it is what caught it. On this dataset:

- query-to-caption cosines average **0.904**
- query-to-image cosines average **0.307**

Same nominal range, ~3× difference in typical magnitude. This is CLIP's **modality gap**: a
known artifact of the contrastive objective, which only ever pulls *matched* pairs
together and never calibrates the two distributions against one another. Text and image
embeddings end up in separate cones of the space.

In a weighted sum, the bigger numbers win regardless of what the weight says. Recall@10
fell off a cliff — 0.592 at `w=0.2` down to 0.406 at `w=0.3` — as the caption magnitudes
swamped the image ranking entirely.

The fix is to throw the scores away and keep only the **positions**:

```
score = w / (k + rank_in_text) + (1 − w) / (k + rank_in_image)        k = 60
```

This is **Reciprocal Rank Fusion**. Being 1st matters; being 1st *by how much* is
discarded. Rank is scale-free, so a 3× magnitude difference can't distort it. The cliff
disappears and Recall@10 holds near 0.59 across every sensible weight.

Recall@10 went **0.397 → 0.599** on that one change. Nothing about the model or the
indexes changed; only the arithmetic combining two lists.

A side benefit falls out for free: a meme found by only one index earns just that index's
weighted contribution, while a meme both indexes rank highly earns both. Agreement between
the two views becomes a tiebreaker without a line of code spent on it.

## Step 5: serve it

```
query → CLIP text encoder → one 512-d vector
      → search text.index  (top 20)
      → search image.index (top 20)
      → fuse the two rankings → keep top 5
      → id_map: row number → meme id
      → one batched MongoDB query → paths and captions
      → JSON
```

Both indexes are oversampled 4× before fusion, so re-ranking has candidates to work with
rather than being trapped inside each index's own top 5.

## What it costs, and the surprise

A hybrid query takes about **29 ms** end to end. The split is the counter-intuitive part:

| Stage | Time | Share |
|---|---|---|
| CLIP text encoding | ~28 ms | ~99% |
| FAISS search, both indexes | 0.30 ms | ~1% |

The vector search — the part that sounds like it should be expensive, the part every
tutorial optimises — is **1% of the cost**. Encoding the query is 95× more expensive than
searching 14,000 vectors for it.

This has a direct consequence for what's worth doing next. Swapping FAISS's exact
brute-force scan for an approximate index (IVF, HNSW) is the standard advice, and here it
would optimise the 0.30 ms leg and leave the 28 ms leg untouched — trading exactness for
nothing. Caching query embeddings attacks the part that actually costs something. That's
why the roadmap is ordered the way it is, and it's only knowable because it was measured.

Exact search is also simply the right call at this size: 6,989 vectors scanned brute-force
is sub-millisecond and **lossless**, with no training step and no tuning parameters. It
becomes the wrong call somewhere past a million vectors, where O(n) per query stops being
free.

## Where to go next

- [ARCHITECTURE.md](ARCHITECTURE.md) — data model, the invariant linking the two indexes,
  component choices argued against their alternatives
- [EVALUATION.md](EVALUATION.md) — how the numbers above were measured, the full weight
  sweep, and the relevance judgments that *didn't* replicate
- [LIMITATIONS.md](LIMITATIONS.md) — what this doesn't do
- [`services/search.py`](../services/search.py) — the query path, top to bottom
- [`services/fusion.py`](../services/fusion.py) — the fusion maths, dependency-free
