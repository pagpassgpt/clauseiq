# ClauseIQ v7 Apex Research Release

## Verified locally

- 12/12 automated tests pass.
- Synthetic benchmark: 155 queries, 72 clauses, 80-query frozen holdout.
- Holdout Recall@5: 0.9500.
- Holdout MRR: 0.7244 (bootstrap 95% CI: 0.6513–0.7935).
- Holdout nDCG@5: 0.7771.
- p50 retrieval latency: ~0.99 ms; p95: ~1.13 ms in the local benchmark environment.
- Benchmark integrity gate: PASS.
- Red-team gate: 3/3 PASS.

## External benchmark status

The release contains executable adapters for ACORD and MLEB/MTEB-style JSONL datasets. The execution environment used to produce this release did not have network access to download the third-party datasets, so **no external score is fabricated**.

ACORD is the primary external contract-clause retrieval benchmark. MLEB adds a broader legal embedding evaluation layer. CUAD provides expert clause/evidence annotations. Published third-party model results are stored as reference metadata and are explicitly not attributed to ClauseIQ.

## Research claim boundary

The strongest defensible claim is that ClauseIQ has a reproducible, frozen internal benchmark and a research-grade external evaluation harness. A claim of superiority on external legal benchmarks should only be made after running the exact external dataset and recording the resulting artifact.
