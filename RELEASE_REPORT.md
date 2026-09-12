# ClauseIQ v6 Release Report

## Release posture

**Goal:** research-grade contract intelligence with production-style engineering discipline.

### Verified locally

- Python test suite: **12/12 passing** before v6 evaluation extensions.
- Frozen retrieval holdout Recall@5: **95.0%**.
- Frozen holdout MRR: **0.7244**.
- Frozen holdout nDCG@5: **0.7771**.
- Exact grounding: **100%** on deterministic quality probes.
- Unsupported-evidence rejection: **100%** on deterministic quality probes.
- Risk determinism: **100%**.
- Agent trajectory benchmark: **100% mean gold retrieval recall** on the included trajectory cases, 0 failed tool calls.
- New benchmark-integrity and red-team gates included in the release.

## What is deliberately not claimed

- No claim of legal advice or legal accuracy.
- No ACORD/CUAD leaderboard score is invented.
- No claim that optional dense/neural components beat the frozen sparse baseline without an actual external evaluation.
- Synthetic benchmark results are regression measurements only.

## Research-grade next experiment

Run ACORD with the external evaluator, then compare:

1. BM25
2. ClauseIQ hybrid sparse retrieval
3. optional dense retrieval
4. optional legal-domain neural retrieval/reranking
5. hybrid + reranking

Report MRR, Recall@k, nDCG@k, latency, bootstrap intervals and failure categories. ACORD's expert qrels make this substantially more credible than scaling up a synthetic dataset. citeturn0search1

## Why this is stronger than a single benchmark score

The release treats a retrieval score as only one component of system quality. Evidence must independently ground to the source clause; risk is deterministic; agent actions are bounded and auditable; and the benchmark includes leakage and adversarial gates.
