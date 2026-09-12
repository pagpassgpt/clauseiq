# ClauseIQ v7 — Evidence-First Contract Intelligence

ClauseIQ is a research/production-style contract intelligence system built to answer a harder question than “can an LLM summarize a contract?”:

> **Can every material claim be retrieved, grounded, policy-checked, audited, and evaluated independently?**

The system combines clause-aware document parsing, hybrid retrieval, bounded tool use, evidence verification, deterministic risk controls, persistence, audit trails, contract comparison, and reproducible evaluation.

## What is implemented

- PDF, DOCX, TXT and Markdown ingestion.
- Clause-aware chunking with source offsets.
- Contract-scoped word TF-IDF, character TF-IDF and BM25 retrieval.
- Development-tuned fusion with a frozen holdout protocol.
- Conservative legal query expansion; original user wording is retained.
- Optional dense retrieval via Sentence Transformers.
- Optional neural reranking; deterministic fallback remains available.
- Evidence object model: `clause_id + quote + offsets + grounding status`.
- Exact/partial/unsupported grounding verification before persistence.
- Deterministic policy and risk scoring independent of LLM output.
- Bounded agent tool loop with explicit schemas and contract scope.
- Tools: search, clause lookup, policy lookup, risk calculation, finding recording and contradiction search.
- Multi-intent query handling and abstention behavior.
- Persistent contracts, clauses, findings, reviews and audit events.
- Deterministic contract comparison by review-relevant categories.
- JSON export endpoint for downstream analysis.
- React/Vite review dashboard with findings, evidence and audit views.
- Unit, integration, adversarial, security-limit and trajectory tests.
- Synthetic regression benchmark plus external BEIR/ACORD evaluation runner.
- Leakage/integrity checks, bootstrap confidence intervals and latency measurements.

## Architecture

```text
                       +-----------------------+
                       | PDF/DOCX/TXT/Markdown |
                       +-----------+-----------+
                                   |
                         Parse / normalize
                                   |
                         Clause segmentation
                                   |
                 +-------------------------------+
                 | Contract-scoped index         |
                 | word TF-IDF / char TF-IDF /   |
                 | BM25 / optional dense vectors |
                 +---------------+---------------+
                                 |
                       adaptive retrieval
                                 |
                 +---------------+---------------+
                 | bounded review agent          |
                 | search / clause / policy /    |
                 | risk / finding / contradiction|
                 +---------------+---------------+
                                 |
                    independent evidence gate
                                 |
             +-------------------+-------------------+
             |                                       |
       grounded findings                      deterministic risk
             |                                       |
             +-------------------+-------------------+
                                 |
                     review + audit persistence
                                 |
                      API / dashboard / export
```

## Evaluation philosophy

ClauseIQ deliberately separates:

1. **Retrieval** — Recall@k, MRR, nDCG and latency.
2. **Grounding** — whether cited evidence exists in the referenced clause.
3. **Safety** — unsupported evidence rejection, scope isolation and adversarial behavior.
4. **Risk policy** — deterministic repeatability.
5. **Agent behavior** — tool correctness, failures, trajectory length and abstention.

This avoids hiding a retrieval regression behind a high end-to-end score.

### Internal synthetic benchmark

The current frozen synthetic benchmark contains 72 clauses, 155 queries and an 80-query holdout. The v7 protocol tunes fusion weights only on development data.

Current frozen results:

| Metric | Holdout |
|---|---:|
| Recall@1 | 56.25% |
| Recall@3 | 85.00% |
| Recall@5 | **95.00%** |
| MRR | **72.44%** |
| nDCG@5 | **77.71%** |
| p50 latency | ~1 ms |
| p95 latency | ~1.2 ms |

These are **synthetic engineering-regression results, not legal accuracy claims**.

### External legal benchmark

The recommended independent retrieval benchmark is [ACORD](https://huggingface.co/datasets/theatticusproject/acord), which contains 114 attorney-written queries and more than 126,000 query-clause pairs with expert relevance ratings. It is distributed in BEIR format with lawyer-derived relevance grades.

[CUAD](https://www.atticusprojectai.org/cuad) is used as a complementary evidence/extraction benchmark: 510 commercial contracts, 13,000+ expert labels and 41 clause categories.

ClauseIQ does not bundle those third-party datasets.

ClauseIQ does not bundle those third-party datasets.

Run an external benchmark after downloading the official dataset:

```bash
PYTHONPATH=backend python -m eval.external_beir /path/to/acord --split test
```

The runner reports graded nDCG, MRR, Recall@5/10, precision at relevance thresholds, latency and bootstrap confidence intervals.

## Reproduce everything

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest

PYTHONPATH=backend python -m eval.benchmark_v5
PYTHONPATH=backend python -m eval.quality_benchmark_v3
PYTHONPATH=backend python -m eval.agent_eval
PYTHONPATH=backend python -m eval.benchmark_integrity
PYTHONPATH=backend python -m eval.red_team
```

Run the API:

```bash
uvicorn clauseiq.main:app --reload --app-dir backend
```

Open `/docs` for the API contract.

## Model boundary

The default path is deterministic/offline. An OpenAI-compatible endpoint can be configured for model experiments. NVIDIA-compatible configuration is supported through environment variables; model output is never granted authority over evidence grounding or deterministic risk policy.

For dense retrieval:

```bash
pip install -e ".[ml]"
```

For research experiments, a legal-domain retriever can be supplied through the same optional retrieval boundary without changing the evidence/risk architecture.

## Security model

Contract text is **data, not instructions**. Retrieved text is passed to the model as untrusted evidence. It cannot redefine tools, grant permissions, access another contract, or change the risk policy.

Every persisted finding must pass independent quote verification. The API enforces upload limits, contract-scoped access and bounded agent steps.

See `SECURITY.md`.

## Research claims policy

The repository never claims “legal accuracy” from synthetic data. External results must state the exact dataset version, split, retrieval configuration, timestamp, confidence interval and failures. Test data is never used for tuning.

That distinction is intentional: the project is designed to demonstrate **measurement discipline**, not benchmark gaming.

## Apex research matrix

The release includes a benchmark registry at `eval/research_matrix.json` and a unified evaluator at `eval/external_suite.py`. The intended final evidence ladder is: frozen internal holdout → ACORD expert retrieval → MLEB contractual retrieval → CUAD evidence extraction → end-to-end legal RAG.

The repository records published external reference scores separately from ClauseIQ measurements. In particular, a published Legal-ColBERT model reports 0.8374 NDCG@10 on MLEB Contractual Clause Retrieval; this is an external reference, not a ClauseIQ result.
