# Retrieval evaluation

Ranking quality used to be assessed by eye. It is now measured. This document covers how
the eval set is built, what the numbers are, what they justified changing, and — just as
importantly — what they do **not** establish.

Reproduce everything here with:

```bash
python scripts/build_eval_set.py && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/evaluate_retrieval.py
```

## Method

**Ground truth** comes from `data/labels.csv`, the source dataset's *human-corrected*
captions — a column this project never indexes. The search indexes embed our own EasyOCR
output instead, and the two disagree for **93%** of memes, with ordinary OCR damage
(`1oyears` for `10 years`, dropped and reordered tokens). A query drawn from the corrected
caption is therefore not a lookup of the indexed string.

**Queries** are a random 6-word contiguous window of a corrected caption, which models
partial recall ("one does not simply") rather than a full transcript. Memes with fewer than
10 caption words are excluded so a window is a fragment, not the whole thing. 1,000 queries,
seed 42, from 4,862 eligible memes. Only **27.7%** of the resulting queries appear verbatim
in our indexed OCR text, so ~72% require the embedding to bridge OCR noise rather than
match a string.

**Metric choice.** There is exactly one relevant meme per query, so Recall@k, Hit-rate@k
and Precision@k·k all coincide; only Recall@k is reported, plus MRR for rank sensitivity.
Retrieval depth is 40 per index, matching the production oversample.

## Results

1,000 queries, 6,989 memes per index.

| mode | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `text` (OCR only) | 0.216 | 0.312 | 0.351 | 0.262 |
| `image` (visual only) | 0.334 | 0.505 | 0.562 | 0.412 |
| weighted sum, `w=0.5` *(former default)* | 0.278 | 0.377 | 0.397 | 0.327 |
| **RRF, `w=0.2`** *(current default)* | 0.309 | **0.526** | **0.599** | 0.407 |
| RRF, `w=0.1` *(best MRR)* | 0.333 | 0.523 | 0.582 | **0.421** |

Full weight sweep for both fusion strategies:

| w | sum R@10 | sum MRR | RRF R@10 | RRF MRR |
|---|---|---|---|---|
| 0.0 | 0.562 | 0.413 | 0.562 | 0.413 |
| 0.1 | 0.592 | 0.398 | 0.582 | 0.421 |
| 0.2 | 0.592 | 0.400 | 0.599 | 0.407 |
| 0.3 | **0.406** | 0.329 | 0.594 | 0.396 |
| 0.4 | 0.398 | 0.328 | 0.592 | 0.391 |
| 0.5 | 0.397 | 0.327 | 0.596 | 0.377 |
| 0.6 | 0.396 | 0.321 | 0.396 | 0.316 |
| 0.8 | 0.396 | 0.318 | 0.395 | 0.304 |
| 1.0 | 0.351 | 0.268 | 0.351 | 0.268 |

## What the numbers changed

**1. The project was searching the weaker index.** Before hybrid retrieval existed, the
only index queried was the OCR-text one: MRR 0.262. Pure image search scores MRR 0.412 on
the same queries — **57% better** — and the current default reaches 0.407 MRR and 0.599
Recall@10, against 0.351 for the original behaviour (**+71% Recall@10**).

**2. CLIP's image encoder out-reads our OCR pipeline.** This was the surprise. The
prediction was that an eval built from captions would favour the text index. It does the
opposite, because CLIP's image encoder perceives the caption *as rendered pixels* and is
famously sensitive to text in images, while the text index can only ever see EasyOCR's
lossy transcription. Two stages of loss (OCR errors, then embedding) beat one. The
practical reading: for text-bearing images, a cross-modal encoder can be a better reader
than a dedicated OCR stack feeding a text encoder.

**3. The fusion maths was wrong.** The weighted sum assumed the two cosine scores were
comparable because both lie in `[-1, 1]`. They are not: text-to-text cosines average 0.904
against 0.307 for text-to-image — CLIP's modality gap. The sum is thus dominated by text
magnitudes, visible as a cliff where Recall@10 collapses from 0.592 at `w=0.2` to 0.406 at
`w=0.3`. Switching to Reciprocal Rank Fusion, which compares positions rather than
magnitudes, removes the cliff: Recall@10 holds near 0.59 across `w = 0.1 … 0.5`. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the derivation.

**4. `text_weight` has a measured default.** It was 0.5 by assumption and is now **0.2**,
chosen for the best Recall@5 and Recall@10 — the metrics that matter for a results grid —
and for sitting mid-plateau rather than at its edge, which is more robust than the
marginally higher MRR at `w=0.1`.

## What these numbers do not establish

- **This is known-item retrieval, not relevance.** Every query has exactly one correct
  answer, and the task is "find the specific meme whose caption went roughly like this".
  Real users also issue open queries — "something relatable about Mondays" — where many
  memes are acceptable. Nothing here measures that, and MRR 0.41 should not be read as
  "41% of searches are good".
- **Only text-bearing memes are evaluated.** Eligibility required a 10-word caption, so
  memes carrying little or no text are absent from the eval entirely — despite being
  exactly the case the image index was added to serve.
- **The default is tuned on this task.** `w=0.2` is optimal for known-item retrieval. An
  open-ended relevance set could well prefer a different weight, and the two have not been
  reconciled.
- **Single dataset, single model.** 6,989 memes, CLIP ViT-B/32. None of this is known to
  transfer to a larger corpus or a different checkpoint.

## Next

A human-judged relevance set of open-ended queries, so the two task types can be tuned
against each other instead of one standing in for both. That is the honest remaining gap.
