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

## Part 2: open-ended relevance

Known-item retrieval answers "find the meme whose caption went like this". It says
nothing about "something relatable about Mondays", where many memes are acceptable
answers. That needs graded relevance judgments.

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/build_relevance_pool.py
python scripts/build_contact_sheets.py --out /tmp/sheets   # one sheet per query
python scripts/evaluate_relevance.py
```

**Method.** 16 open-ended queries spanning four deliberately different kinds: mood
(`monday morning`), topic (`programming bug`), visual/template (`surprised cat`, where
the caption may say nothing useful), and phrase-like (`one does not simply`, where OCR
text should matter most). Candidates are pooled TREC-style: the top 5 from each of five
configurations, unioned, giving 156 query-meme pairs (9.8 per query). Each pair is graded
**0 / 1 / 2** — irrelevant, partially relevant, relevant. Final distribution: 48% / 24% /
28%.

**Candidates are judged blind.** `build_relevance_pool.py` shuffles the pool with a fixed
seed and records no system attribution, so the judge cannot tell which mode produced which
candidate and cannot flatter a preferred system. The rankings are kept separately for
scoring.

### Results

16 queries, nDCG@5 with exponential gain, P@5 counting only fully relevant results.

| mode | nDCG@5 | P@5 |
|---|---|---|
| `text` (OCR only) | 0.465 | 0.338 |
| `image` (visual only) | 0.884 | 0.637 |
| RRF, `w=0.1` | 0.894 | 0.675 |
| **RRF, `w=0.2`** *(current default)* | **0.896** | **0.675** |
| RRF, `w=0.5` | 0.742 | 0.562 |

**The two evaluations agree on the default.** This was not a given — the worry when tuning
`text_weight` on known-item retrieval was that a task with one right answer per query would
prefer a different weight from open-ended relevance. It does not: `w=0.2` wins both, and
both penalise `w=0.5` and pure `text` the same way. The 0.2 default is therefore supported
by two independent tasks rather than fitted to one.

The image index also dominates here, just as it did on known-item retrieval (0.884 vs
0.465), so that finding is not an artefact of queries having been derived from captions.

### Where it fails

Per-query nDCG@5 at the default, worst first:

| nDCG@5 | query |
|---|---|
| 0.711 | pointing at a screen |
| 0.744 | surprised cat |
| 0.805 | this is fine |
| 0.815 | food craving |
| 0.844 | crying while smiling |
| … | … |
| 1.000 | awkward silence, exam stress, monday morning, one does not simply, public transport |

The pattern is legible: the weakest queries name a **visual template or pose** rather than
a subject. `this is fine` should return the dog in the burning room; it returns Disaster
Girl and Bad Luck Brian — thematically adjacent calamity, but not the template asked for.
`pointing at a screen` returns things that contain screens or pointing, rarely both. Six
queries, mostly concrete subjects or exact phrases, score a perfect 1.000.

Two incidental findings from judging: the dataset contains **near-duplicate memes**
(`meme_002323` and `meme_003828` are the same image), and a handful of generic reaction
memes act as **hubs** — `meme_003827` ("I'll be right back") was pooled for six unrelated
queries and judged irrelevant for most, which is the signature of a vector close to the
centroid of the embedding space.

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

### Specific to the relevance set

- **The judgments did not replicate.** The 156 judgments in
  `data/relevance_pool.jsonl` were made by Claude (recorded in each row's `judged_by`), and
  a second annotator on a 48-pair blind sample agreed at **kappa 0.140** — poor. The
  relevance numbers are therefore one annotator's view, not a measured consensus. See the
  agreement section above.
- **Pooled nDCG is inflated in absolute terms.** The ideal ranking is drawn only from
  pooled documents, all contributed by the systems being compared, so a relevant meme that
  no configuration retrieved is invisible and counts against nobody. nDCG@5 0.896 means
  "near-optimal ordering of what these systems found", not "90% perfect". **P@5 is the
  more interpretable figure: about two of every five top results are fully relevant, with
  roughly another one partially so.** The comparison between modes is sound; the absolute
  level is optimistic.
- **16 queries is a small sample.** Differences of a point or two of nDCG — `w=0.1` versus
  `w=0.2`, for instance — are well inside the noise. No significance testing was done.

## Inter-annotator agreement: the relevance judgments did not replicate

A second annotator independently graded a blind 48-pair sample — three per query, with the
first-pass grades withheld. **Agreement was poor.**

| | |
|---|---|
| Items compared | 48 |
| Exact agreement | 47.9% (23/48) |
| Agreement within one grade | 75.0% |
| **Cohen's kappa** | **0.140** (poor) |
| Quadratic-weighted kappa | 0.091 (poor) |

Confusion matrix (rows: first pass, columns: second annotator):

| | irrelevant | partial | relevant |
|---|---|---|---|
| **irrelevant** | 4 | 1 | 4 |
| **partial** | 2 | 4 | 5 |
| **relevant** | 8 | 5 | 15 |

A kappa of 0.14 is barely above chance. Raw agreement of 48% looks less alarming than it
is: 48% of the pool is graded "relevant", so two annotators agree that often by labelling
everything "relevant".

**Consequence: treat every nDCG@5 and P@5 figure in Part 2 as indicative only.** They
describe one annotator's notion of relevance, and a second annotator did not reproduce it.
The relative ordering of modes may well survive — the gap between `text` (0.465) and
`image` (0.884) is far larger than any plausible judgment noise — but the absolute values
should not be quoted as quality measurements.

**What Part 1 does not depend on.** The known-item evaluation uses the dataset's own
human-corrected captions as ground truth, not these judgments. Recall@k, MRR, the modality
gap finding and the `text_weight = 0.2` default all rest on that objective ground truth and
are unaffected by this result. What is lost is the independent corroboration Part 2 was
providing for the default, not the default itself.

**One observation for anyone reading the raw data**, recorded because it bears on how to
interpret the figure rather than to explain it away: 12 of the 25 disagreements are
outright 0-vs-2 flips rather than arguments about the middle grade, and four items graded
irrelevant by the second annotator have the query's own wording in their caption (item 38,
*"even Dr Evil is taking public transit"*, graded irrelevant to `public transport`). Both
grade sets are published as entered in `data/judgment_sample.csv` and
`data/relevance_pool.jsonl`; recompute with `scripts/compare_judgments.py`.

## Re-judging the sample

The sample and tooling are in the repo and can be re-run with any annotator.

```bash
python scripts/build_judgment_sample.py
python scripts/build_contact_sheets.py --pool data/judgment_sample.jsonl --out sheets/
# fill the `relevance` column in data/judgment_sample.csv with 0, 1 or 2
python scripts/compare_judgments.py
```

48 pairs, three per query so every query type is covered, grouped by query because a grade
only means anything against the query it was judged for. **The first-pass grades are
withheld from the CSV**, so the second annotator is not anchored by them; the sheet cell
labels match the CSV's `item` column.

`compare_judgments.py` reports exact agreement, agreement within one grade, Cohen's kappa
(plain and quadratic-weighted — grades are ordinal, so a 1-vs-2 disagreement should not
count the same as 0-vs-2), a confusion matrix, and every disagreeing item so the
disagreement can be read rather than just counted. It needs no model, index or database.

Read the result like this: raw agreement alone flatters any sample where one grade
dominates — 48% of this pool is graded "relevant", so two annotators agree that often by
chance. Kappa is the figure to quote. Conventionally, below 0.2 is poor, 0.4-0.6 moderate,
0.6-0.8 substantial, above 0.8 near-perfect. A kappa in the moderate range or below would
mean the nDCG figures above rest on one annotator's idiosyncratic notion of relevance and
should be treated as indicative only.

## Next

1. **Adjudicate the 25 disagreements** item by item and settle a written relevance
   guideline, then re-judge against it. Kappa 0.140 says the two annotators were not
   applying the same standard; a guideline is how that gets fixed, and it is a precondition
   for the relevance numbers meaning anything.
2. **Grow the relevance set** past 16 queries so differences between neighbouring weights
   can be tested rather than eyeballed.
3. **Fix the template-query failure.** The weakest queries name a visual template
   (`this is fine`). Template identity is a different retrieval problem from subject
   matter, and may want its own signal rather than a blend weight.
