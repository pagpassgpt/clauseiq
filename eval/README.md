# ClauseIQ evaluation suite

ClauseIQ separates **retrieval quality**, **grounding correctness**, and **deterministic safety**. The benchmark never asks an LLM to grade itself.

## Benchmark v2

```bash
PYTHONPATH=backend python eval/benchmark_v2.py
PYTHONPATH=backend python eval/quality_benchmark.py
```

The retrieval benchmark uses a fixed seed, 24 contract-risk topics, 56 clauses, paraphrased queries, near-neighbor distractors, and a deterministic holdout split. It compares TF-IDF, BM25, normalized hybrid fusion, and reciprocal-rank fusion. It reports Recall@1/3/5, MRR, nDCG@5, p50/p95 latency, and a paired bootstrap 95% interval.

The quality benchmark performs 500 exact-grounding checks, 500 unsupported-evidence rejection checks, and 500 deterministic risk checks. These are regression gates, not model-quality claims.

## Interpreting results

The synthetic benchmark is deliberately published with its generator and labels. Do not copy generated numbers into marketing material. For a stronger research claim, replace or augment the synthetic set with a blind, human-reviewed contract corpus and preserve the same evaluation protocol.
