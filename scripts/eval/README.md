# Evaluation harness

**Nothing here runs at query time.** These seven scripts exist only to produce the numbers
reported in [docs/EVALUATION.md](../../docs/EVALUATION.md). If you are reading the repo to
understand how search works, skip this folder — `services/search.py` and
`services/fusion.py` are the product.

Two separate evaluations live here.

**Known-item retrieval** — 1,000 queries with one correct answer each, built from the
dataset's own human-corrected captions. This is where Recall@10, MRR and the measured
`text_weight = 0.2` default come from:

```bash
python scripts/eval/build_eval_set.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/eval/evaluate_retrieval.py
```

**Open-ended relevance** — 16 broad queries, TREC-style pooled and graded by hand. These
numbers did not replicate between annotators (Cohen's kappa 0.140), so treat them as
indicative only:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/eval/build_relevance_pool.py
python scripts/eval/build_contact_sheets.py --out sheets/   # judge these by hand
python scripts/eval/evaluate_relevance.py
```

`build_judgment_sample.py` and `compare_judgments.py` measure agreement between two
annotators on a blind 48-pair subsample — that is what produced the kappa figure. The
re-judging walkthrough is in
[docs/EVALUATION.md](../../docs/EVALUATION.md#re-judging-the-sample).
