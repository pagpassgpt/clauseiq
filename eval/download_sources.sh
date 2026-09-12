#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/external
cat <<'EOF'
Download the official datasets manually or with your preferred dataset client:
  ACORD: https://huggingface.co/datasets/theatticusproject/acord
  CUAD:  https://huggingface.co/datasets/theatticusproject/cuad
Then place ACORD's corpus.jsonl, queries.jsonl and qrels/test.tsv under data/external/acord/.
This repository does not redistribute third-party datasets.
EOF
