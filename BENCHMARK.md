# ClauseIQ Benchmark Protocol

## Primary internal regression benchmark

- 72 indexed clauses
- 31 topics
- 155 queries
- 42 development queries
- 33 internal test queries
- 80 frozen holdout queries
- deterministic seed: `20260912`

Fusion weights are selected only on development data. The holdout is not used for model selection.

## Frozen holdout result

| Metric | Result |
|---|---:|
| Recall@1 | 0.5625 |
| Recall@3 | 0.8500 |
| Recall@5 | 0.9500 |
| MRR | 0.7244 |
| nDCG@5 | 0.7771 |
| Bootstrap 95% CI for MRR | [0.6513, 0.7935] |

Latency is measured locally and is not a cloud SLA.

## Safety gates

- exact grounding rate: 1.0
- unsupported rejection rate: 1.0
- deterministic risk rate: 1.0
- adversarial scope/evidence gate: required
- benchmark leakage check: required

## External evaluation

ACORD is the primary independent retrieval benchmark. It contains 114 expert-written queries and 126,662+ query-clause pairs rated by lawyers on a 1–5 star scale, represented as 0–4 relevance qrels in BEIR format. citeturn0search1

CUAD is complementary for clause/evidence extraction and contains 510 contracts, 13,000+ expert labels and 41 categories. citeturn0search0

No external score is claimed until the official data are actually evaluated.

## Anti-gaming rules

1. No tuning on test/holdout qrels.
2. No cherry-picking a single query family.
3. Report latency separately from quality.
4. Report confidence intervals where appropriate.
5. Keep negative experiments in the release report.
6. Separate retrieval, grounding, policy, safety and agent metrics.
7. Preserve dataset/license provenance.
