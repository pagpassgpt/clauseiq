"""Reproducible retrieval benchmark for ClauseIQ.

No LLM calls are required. The benchmark compares lexical, BM25, TF-IDF,
hybrid score fusion, and hybrid + deterministic lexical reranking on a small
hand-labelled contract-query set. Metrics are computed from the labels.
"""
from __future__ import annotations
import json, statistics, time
from dataclasses import dataclass
from pathlib import Path
from clauseiq.documents import Clause
from clauseiq.retrieval import HybridRetriever, toks

@dataclass(frozen=True)
class Case:
    query: str
    relevant: frozenset[str]

CASES = [
    Case("uncapped liability exposure", frozenset({"c1"})),
    Case("supplier indemnifies customer for third party claims", frozenset({"c2"})),
    Case("terminate agreement for convenience", frozenset({"c3"})),
    Case("automatic renewal after initial term", frozenset({"c4"})),
    Case("late payment interest", frozenset({"c5"})),
    Case("customer personal data breach notification", frozenset({"c6"})),
    Case("confidential information survives termination", frozenset({"c7"})),
    Case("governing law and exclusive jurisdiction", frozenset({"c8"})),
    Case("service availability uptime commitment", frozenset({"c9"})),
    Case("intellectual property ownership of deliverables", frozenset({"c10"})),
    Case("audit rights for compliance", frozenset({"c11"})),
    Case("assignment requires consent", frozenset({"c12"})),
]

TEXT = [
("c1", "1", "LIABILITY. Supplier's liability shall be unlimited and without limitation for all claims arising under this Agreement."),
("c2", "2", "INDEMNITY. Supplier shall indemnify, defend, and hold harmless Customer from third-party claims arising from Supplier's services."),
("c3", "3", "TERMINATION. Customer may terminate this Agreement for convenience on thirty days' written notice without cause."),
("c4", "4", "RENEWAL. This Agreement shall automatically renew for successive one-year terms unless either party gives notice."),
("c5", "5", "PAYMENT. Overdue invoices accrue interest at 1.5 percent per month until paid in full."),
("c6", "6", "DATA SECURITY. Supplier shall notify Customer of any personal data breach within twenty-four hours of discovery."),
("c7", "7", "CONFIDENTIALITY. Confidentiality obligations survive termination for five years after this Agreement ends."),
("c8", "8", "GOVERNING LAW. This Agreement is governed by the laws of Delaware and the parties submit to its exclusive jurisdiction."),
("c9", "9", "SERVICE LEVELS. Supplier guarantees 99.9 percent monthly service availability, excluding scheduled maintenance."),
("c10", "10", "INTELLECTUAL PROPERTY. Customer owns all right, title, and interest in deliverables specifically created and paid for by Customer."),
("c11", "11", "AUDIT. Customer may audit Supplier's relevant records once per year to verify compliance with this Agreement."),
("c12", "12", "ASSIGNMENT. Neither party may assign this Agreement without the other party's prior written consent, except in a merger."),
# distractors
("d1", "13", "NOTICES. Notices shall be delivered by email or recognized overnight courier to the addresses designated by each party."),
("d2", "14", "FORCE MAJEURE. Neither party is responsible for delay caused by events beyond its reasonable control."),
("d3", "15", "DEFINITIONS. Business Day means any day other than Saturday, Sunday, or a public holiday."),
("d4", "16", "ENTIRE AGREEMENT. This document constitutes the entire agreement and supersedes prior discussions."),
]

def corpus(): return [Clause(i, s, t, 0, len(t)) for i,s,t in TEXT]

def precision_at(ids, relevant, k):
    got=ids[:k]; return sum(x in relevant for x in got)/k

def recall_at(ids, relevant, k):
    return sum(x in relevant for x in ids[:k])/len(relevant)

def mrr(ids, relevant):
    for rank, cid in enumerate(ids, 1):
        if cid in relevant: return 1.0/rank
    return 0.0

def ndcg(ids, relevant, k):
    import math
    dcg=sum((1/math.log2(i+2)) for i,c in enumerate(ids[:k]) if c in relevant)
    ideal=sum((1/math.log2(i+2)) for i in range(min(k,len(relevant))))
    return dcg/ideal if ideal else 0.0

def lexical_rank(clauses, query):
    q=set(toks(query)); return sorted(clauses, key=lambda c: (len(q & set(toks(c.text))), c.id), reverse=True)

def run():
    clauses=corpus(); retr=HybridRetriever(clauses)
    methods={"tfidf":lambda q: sorted(clauses,key=lambda c: retr.vectorizer.transform([q]).dot(retr.X.T).toarray()[0][clauses.index(c)],reverse=True),
             "bm25":lambda q: sorted(clauses,key=lambda c: retr._bm25(q)[clauses.index(c)],reverse=True),
             "hybrid":lambda q: [next(c for c in clauses if c.id==h.clause_id) for h in retr.search(q,len(clauses))],
             "rrf":lambda q: [next(c for c in clauses if c.id==h.clause_id) for h in retr.search_rrf(q,len(clauses))],
             "hybrid_lexical_rerank":lambda q: lexical_rank([next(c for c in clauses if c.id==h.clause_id) for h in retr.search(q,len(clauses))],q)}
    out={}
    for name,fn in methods.items():
        rows=[]; lat=[]
        for case in CASES:
            t=time.perf_counter(); ids=[c.id for c in fn(case.query)]; lat.append((time.perf_counter()-t)*1000)
            rows.append({"query":case.query,"mrr":mrr(ids,case.relevant),"r@1":recall_at(ids,case.relevant,1),"r@3":recall_at(ids,case.relevant,3),"ndcg@5":ndcg(ids,case.relevant,5)})
        out[name]={"mrr":statistics.mean(r["mrr"] for r in rows),"recall@1":statistics.mean(r["r@1"] for r in rows),"recall@3":statistics.mean(r["r@3"] for r in rows),"ndcg@5":statistics.mean(r["ndcg@5"] for r in rows),"p50_ms":statistics.median(lat),"cases":len(rows)}
    return out

if __name__=="__main__":
    result=run(); print(json.dumps(result,indent=2))
    Path(__file__).with_name("benchmark_results.json").write_text(json.dumps(result,indent=2)+"\n")
