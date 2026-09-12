"""Research-grade retrieval benchmark with dev tuning and a frozen holdout.

Protocol:
- synthetic contract topics are used only to validate engineering changes;
- the first-stage fusion weights are selected on the development partition;
- the held-out partition is never used to choose weights;
- metrics include Recall@1/3/5, MRR and nDCG@5;
- multi-intent retrieval is evaluated separately;
- results include confidence intervals for the primary holdout MRR.
"""
from __future__ import annotations
import hashlib, json, math, random, statistics, time
from pathlib import Path
from eval.benchmark_v3 import TOPICS, DISTRACTORS, Case, build, score, agg
from clauseiq.documents import Clause
from clauseiq.query import expand_query, split_multi_hop
from clauseiq.retrieval import HybridRetriever


def bootstrap_mean_ci(values, seed=20260912, n=5000):
    rng=random.Random(seed); means=[]
    for _ in range(n):
        means.append(statistics.mean(rng.choice(values) for _ in values))
    means.sort()
    return [means[int(.025*n)], means[int(.975*n)]]


def rank_with_weights(retriever, query, weights):
    w,c,b=retriever._signals(expand_query(query))
    wn,cn,bn=map(retriever.norm,(w,c,b))
    a,d,e=weights
    vals=[a*wn[i]+d*cn[i]+e*bn[i] for i in range(len(retriever.clauses))]
    return [retriever.clauses[i].id for i in sorted(range(len(vals)), key=lambda i:(-vals[i], retriever.clauses[i].id))]


def split_case(case):
    if case.split == "holdout":
        return "holdout"
    # Deterministic query-level split prevents weight tuning from seeing the reported internal test cases.
    bucket=int(hashlib.sha256(case.query.encode()).hexdigest()[:8],16)%10
    return "dev" if bucket < 6 else "test"


def tune_weights(retriever, cases):
    best=None
    for ai in range(21):
        for bi in range(21-ai):
            weights=(ai/20,bi/20,1-(ai+bi)/20)
            rows=[]
            for case in cases:
                if split_case(case) != 'dev':
                    continue
                rows.append(score(rank_with_weights(retriever,case.query,weights),case.relevant))
            metrics=agg(rows)
            # MRR is primary because ranking the first useful clause is most valuable in review.
            key=(metrics['mrr'],metrics['ndcg@5'],metrics['recall@5'])
            if best is None or key > best[0]:
                best=(key,weights,metrics)
    return {'weights':best[1], 'dev_metrics':best[2]}


def multi_intent_eval(retriever):
    pairs=[
      ("end the deal without cause and privacy incident notification deadline", {"c03","c06"}),
      ("post termination data erasure and confidentiality continues after the contract ends", {"c16","c07"}),
      ("uptime promise and credit for downtime", {"c09","c23"}),
      ("minimum vendor coverage and uncapped direct exposure", {"c13","c01"}),
      ("approval for data processors and where customer data may be stored", {"c15","c26"}),
      ("post contract hiring restriction and source code backup release mechanism", {"c28","c30"}),
      ("annual pricing adjustment limit and late invoice finance charge", {"c29","c05"}),
      ("performance warranty and service conformity promise", {"c18"}),
    ]
    rows=[]
    for query,gold in pairs:
        # Each intent contributes its strongest hit, then remaining capacity is filled by global ranking.
        parts=split_multi_hop(query)
        selected=[]; seen=set()
        for part in parts[:4]:
            hits=retriever.search(part,3)
            for h in hits:
                if h.clause_id not in seen:
                    selected.append(h.clause_id); seen.add(h.clause_id)
                    break
        for h in retriever.search(query,len(retriever.clauses)):
            if h.clause_id not in seen:
                selected.append(h.clause_id); seen.add(h.clause_id)
            if len(selected)>=5: break
        rows.append(len(set(selected[:5])&gold)/len(gold))
    return {'queries':len(rows),'mean_gold_recall@5':statistics.mean(rows),'all_gold_retrieved_rate':sum(x==1 for x in rows)/len(rows)}


def run():
    clauses,cases=build(); base=HybridRetriever(clauses)
    tuning=tune_weights(base,cases)
    tuned=HybridRetriever(clauses,weights=tuning['weights'])
    rows=[]; lats=[]; dev_rows=[]; test_rows=[]; hold_rows=[]
    for case in cases:
        t=time.perf_counter(); rank=rank_with_weights(base,case.query,tuning['weights']); lats.append((time.perf_counter()-t)*1000)
        row=score(rank,case.relevant)
        (hold_rows if split_case(case)=='holdout' else test_rows if split_case(case)=='test' else dev_rows).append(row)
        rows.append({'query':case.query,'split':case.split,'rank':rank.index(case.relevant)+1 if case.relevant in rank else None})
    hold_mrr=[score(rank_with_weights(base,c.query,tuning['weights']),c.relevant)[3] for c in cases if c.split=='holdout']
    out={
      'version':'6.0',
      'protocol':{'tuning_split':'development only','primary_metric':'MRR','frozen_holdout':True,'weights':tuning['weights']},
      'dataset':{'topics':len(TOPICS),'clauses':len(clauses),'queries':len(cases),'development':sum(split_case(c)=='dev' for c in cases),'internal_test':sum(split_case(c)=='test' for c in cases),'holdout':sum(split_case(c)=='holdout' for c in cases)},
      'development':tuning['dev_metrics'],
      'internal_test':agg(test_rows),
      'holdout':agg(hold_rows),
      'holdout_mrr_bootstrap95':bootstrap_mean_ci(hold_mrr),
      'latency_ms':{'p50':statistics.median(lats),'p95':sorted(lats)[max(0,int(.95*len(lats))-1)]},
      'multi_intent':multi_intent_eval(tuned),
      'note':'Synthetic benchmark; external expert-labeled evaluation is provided as a reproducible runner.'
    }
    Path(__file__).with_name('benchmark_v5_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    return out

if __name__=='__main__': run()
