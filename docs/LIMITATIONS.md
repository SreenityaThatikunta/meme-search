# Known limitations

Worth being straight about these. The README links here rather than burying its quickstart
under caveats.

## Measurement

- **The relevance judgments did not replicate.** A second annotator graded a blind 48-pair
  sample and agreed with the first pass at **Cohen's kappa 0.140 — poor**, barely above
  chance. The nDCG@5 and P@5 figures therefore describe one annotator's notion of relevance
  rather than a measured consensus, and should be read as indicative only. The large gap
  between modes (0.465 for `text` vs 0.884 for `image`) is probably bigger than judgment
  noise; the absolute values are not trustworthy. Both grade sets are published — see
  [inter-annotator agreement](EVALUATION.md#inter-annotator-agreement-the-relevance-judgments-did-not-replicate).
- **This does not affect the known-item numbers.** Recall@k, MRR, the modality-gap finding
  and the `text_weight = 0.2` default come from ground truth in the dataset's own
  human-corrected captions, not from these judgments.
- **Absolute nDCG is optimistic too.** Pooled judging draws the ideal ranking only from what
  these systems retrieved, so 0.896 would mean "near-optimal ordering of what was found"
  even if the judgments were solid.
- **Small samples where it matters.** 16 open-ended queries cannot separate neighbouring
  blend weights; no significance testing was done. The known-item set (1,000 queries) also
  required a 10-word caption, so memes with little or no text are absent from it.
- **Latency is one machine, one configuration.** Measured on an Apple M2 Pro, CPU only,
  with MongoDB excluded because it wasn't running on the benchmark host. It's a retrieval
  figure, not an SLO.

## Retrieval quality

- **Template queries are the weak spot.** Queries naming a visual template rather than a
  subject do worst: `this is fine` returns thematically adjacent calamity memes, not the
  burning-room dog. Template identity is a different retrieval problem from subject matter
  and probably wants its own signal rather than a blend weight.
- **OCR is the pipeline's weak link.** EasyOCR struggles with heavy stylisation and
  low-contrast captions, and the Canny edge pre-filter that speeds the pass up can skip a
  genuinely text-bearing meme. Errors here propagate silently into the text index — a meme
  whose caption was misread is simply unfindable by text.
- **`text_weight` is tuned for one task.** 0.2 is optimal for known-item retrieval; an
  open-ended relevance set could prefer another value, and the two are not reconciled.

## Engineering

- **Exact search won't scale.** `IndexFlatIP` is a brute-force scan — fine and lossless at
  7k vectors (0.30 ms measured), but O(n) per query. Past ~1M vectors this needs
  `IndexIVFFlat` or HNSW, which trades a little recall for sublinear search.
- **The CLIP encoder is the latency floor.** ~28 of every 29 ms goes on encoding the query,
  and that cost is fixed per request with no caching in place.
- **Single process, in-memory indexes.** The model and both indexes load per worker, so
  horizontal scaling multiplies memory rather than sharing it.
- **Open CORS and no auth.** Fine for `localhost`; both need tightening before this is
  exposed anywhere. Set `MEME_CORS_ORIGINS` — see [CONFIGURATION.md](CONFIGURATION.md).
- **Not deployed.** The backend runs locally by design, which is why the Chrome extension
  needs a local API running and there's no hosted demo link.

## Roadmap

Ordered by measured impact, not by what sounds impressive:

1. **Settle a written relevance guideline and re-judge.** Kappa 0.140 says the two
   annotators were not applying the same standard, which is a precondition for the
   open-ended numbers meaning anything.
2. **Cache query embeddings.** The benchmark puts ~99% of latency in the CLIP text encoder,
   so an LRU cache on repeated queries is the single largest available win.
3. **Accept image queries** (upload a meme, find similar) — the image index already
   supports this; it needs only an endpoint and the image preprocessing path.
4. **Deploy the API** so the extension and a demo link work without a local server.
5. **Revisit `IndexIVFFlat`** only past ~1M vectors. At current scale it would trade
   exactness for a speedup on the 0.30 ms leg while leaving the ~28 ms leg untouched.
