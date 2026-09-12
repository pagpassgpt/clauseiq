"""Leakage and benchmark-integrity checks for reproducible evaluation."""
from __future__ import annotations
import hashlib, json, re
from collections import defaultdict
from eval.benchmark_v3 import build
from eval.benchmark_v5 import split_case

def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
def run():
    clauses,cases=build(); seen=defaultdict(list)
    for c in cases: seen[norm(c.query)].append(c.split)
    exact_dupes={q:v for q,v in seen.items() if len(v)>1}
    fingerprints=defaultdict(list)
    for c in cases: fingerprints[hashlib.sha256(norm(c.query).encode()).hexdigest()].append(c.query)
    overlap=[]
    for c in cases:
        if split_case(c)=='holdout': continue
        a=set(norm(c.query).split())
        for h in cases:
            if split_case(h)!='holdout': continue
            b=set(norm(h.query).split())
            j=len(a&b)/max(1,len(a|b))
            if j>=0.9: overlap.append((c.query,h.query,j))
    out={'queries':len(cases),'clauses':len(clauses),'exact_duplicate_queries':len(exact_dupes),'near_duplicate_holdout_pairs':len(overlap),'status':'PASS' if not exact_dupes and not overlap else 'REVIEW'}
    print(json.dumps(out,indent=2)); return out
if __name__=='__main__': run()
