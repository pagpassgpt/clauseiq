# External evaluation: the final credibility gate

ClauseIQ's synthetic benchmark is an engineering regression suite. The final research claim should come from independent expert-labeled data.

## Primary benchmark: ACORD

ACORD contains 114 attorney-written queries and more than 126,000 query-clause pairs with lawyer relevance ratings. It is provided in BEIR format and is specifically designed for contract-clause retrieval.

Official dataset: https://huggingface.co/datasets/theatticusproject/acord

```bash
PYTHONPATH=backend python -m eval.external_suite data/external/acord --split test --out results/acord.json
```

## MLEB contractual clause retrieval

The MLEB contractual-clause task contains 45 clause types with two representative examples each and is intended as a zero-shot legal retrieval stress test.

Official dataset: https://huggingface.co/datasets/isaacus/contractual-clause-retrieval

```bash
PYTHONPATH=backend python -m eval.external_suite data/external/mleb-contractual-clause-retrieval --out results/mleb_clause.json
```

A published third-party reference model, Legal-ColBERT Clause Retriever, reports NDCG@10 0.8374, MAP 0.7777 and Recall@10 0.9444 on that benchmark. These numbers are **not ClauseIQ measurements** and must not be presented as such.

## CUAD

CUAD contains 510 commercial contracts, 13,000+ expert labels and 41 clause categories. It is primarily useful for evidence extraction/grounding rather than being treated as a pure retrieval leaderboard.

Official dataset: https://huggingface.co/datasets/theatticusproject/cuad

## Rules

1. Never tune on the external test split.
2. Pin dataset revisions in the result artifact.
3. Record model name, version, dependencies and hardware.
4. Report confidence intervals and per-category failures.
5. Separate retrieval, grounding, safety and generation metrics.
6. Never convert a published third-party score into a ClauseIQ score.
