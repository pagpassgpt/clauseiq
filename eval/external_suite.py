"""Unified external legal retrieval evaluation.

Supports BEIR-style directories and MTEB/MLEB JSONL layouts. The suite is
method-agnostic and can evaluate sparse ClauseIQ retrieval plus optional dense
retrieval and neural reranking when those dependencies/models are installed.
No third-party dataset is bundled.
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
from clauseiq.documents import Clause
from clauseiq.retrieval import HybridRetriever, OptionalDenseRetriever


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def read_tsv(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lower().startswith("query-id"):
            continue
        q, d, s = line.split("\t")[:3]
        out.setdefault(q, {})[d] = float(s)
    return out

def load_dataset(root: Path, split: str = "test"):
    corpus = root / "corpus.jsonl"
    queries = root / "queries.jsonl"
    qrels = root / "qrels" / f"{split}.tsv"
    if corpus.exists() and queries.exists() and qrels.exists():
        return read_jsonl(corpus), read_jsonl(queries), read_tsv(qrels), "beir"
    # MLEB/MTEB layout: default.jsonl, corpus.jsonl, queries.jsonl.
    default = root / "default.jsonl"
    if corpus.exists() and queries.exists() and default.exists():
        rel = {}
        for row in read_jsonl(default):
            rel.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = float(row.get("score", 1))
        return read_jsonl(corpus), read_jsonl(queries), rel, "mteb"
    raise FileNotFoundError(f"Unsupported external dataset layout at {root}")

def dcg(vals):
    import math
    return sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(vals))

def ndcg(rank, rel, k):
    got = [rel.get(d, 0) for d in rank[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    den = dcg(ideal)
    return dcg(got) / den if den else 0.0

def recall(rank, rel, k, threshold=1):
    gold = {d for d, s in rel.items() if s >= threshold}
    return len(set(rank[:k]) & gold) / len(gold) if gold else 0.0

def precision(rank, rel, k, threshold):
    return sum(rel.get(d, 0) >= threshold for d in rank[:k]) / max(1, k)

def mrr(rank, rel, threshold=1):
    for i, d in enumerate(rank, 1):
        if rel.get(d, 0) >= threshold:
            return 1 / i
    return 0.0

def evaluate(root, split="test", dense_model=None, top_k=10):
    corpus, queries, qrels, fmt = load_dataset(Path(root), split)
    clauses = [Clause(str(x["_id"]), str(x.get("title", "")), str(x.get("text", "")), 0, 0) for x in corpus]
    sparse = HybridRetriever(clauses)
    methods = {"clauseiq_hybrid": sparse}
    if dense_model:
        methods[f"dense:{dense_model}"] = OptionalDenseRetriever(clauses, model_name=dense_model)

    results = {}
    for name, retriever in methods.items():
        rows, lats = [], []
        for q in queries:
            qid = str(q["_id"])
            if qid not in qrels:
                continue
            text = str(q["text"])
            t0 = time.perf_counter()
            hits = retriever.search(text, k=top_k, expand=True) if name == "clauseiq_hybrid" else retriever.search(text, k=top_k)
            lats.append((time.perf_counter() - t0) * 1000)
            rank = [h.clause_id for h in hits]
            rel = qrels[qid]
            rows.append({
                "ndcg@5": ndcg(rank, rel, 5), "ndcg@10": ndcg(rank, rel, 10),
                "mrr": mrr(rank, rel), "recall@5": recall(rank, rel, 5),
                "recall@10": recall(rank, rel, 10), "precision@5_ge3": precision(rank, rel, 5, 3),
                "precision@5_ge4": precision(rank, rel, 5, 4), "precision@5_ge5": precision(rank, rel, 5, 5),
            })
        results[name] = {
            "queries": len(rows),
            "metrics": {k: statistics.mean(r[k] for r in rows) for k in rows[0]} if rows else {},
            "latency_ms": {"p50": statistics.median(lats) if lats else None,
                           "p95": sorted(lats)[max(0, int(.95 * len(lats)) - 1)] if lats else None},
        }
    return {"dataset": str(root), "format": fmt, "split": split, "corpus": len(corpus), "results": results}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--split", default="test")
    p.add_argument("--dense-model", default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    result = evaluate(a.dataset, a.split, a.dense_model, a.top_k)
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
