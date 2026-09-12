# ClauseIQ Research Evaluation Protocol

## External legal retrieval benchmark

The primary external benchmark is **ACORD**, an expert-annotated legal contract retrieval dataset with 114 attorney-written queries and 126,662+ query-clause pairs rated from 1–5 stars. The official dataset is distributed in BEIR format with train/validation/test qrels.

Official source: https://huggingface.co/datasets/theatticusproject/acord

Run:

```bash
PYTHONPATH=backend python -m eval.external_beir /path/to/acord
```

The repository intentionally does **not** redistribute the dataset. Download it from the official source and follow its CC BY 4.0 terms.

## Secondary evidence benchmark

CUAD provides 510 commercial contracts and 13,000+ expert labels across 41 clause categories. It is useful for evaluating clause/evidence extraction and grounding, but it is not itself a pure retrieval benchmark. Keep CUAD evaluation separate from ACORD retrieval evaluation.

Official source: https://huggingface.co/datasets/theatticusproject/cuad

## Claims policy

- Synthetic benchmark numbers are engineering regression tests, not legal accuracy claims.
- External benchmark numbers must include the dataset version, split, command, model configuration and timestamp.
- Never tune on the held-out test split.
- Report failures and confidence intervals, not only the best metric.
- Retrieval, grounding, risk policy and agent trajectory scores remain separate.
