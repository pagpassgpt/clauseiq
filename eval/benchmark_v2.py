"""ClauseIQ benchmark v2: deterministic, held-out-style retrieval evaluation.

The dataset is generated from fixed templates so it is reproducible without
external data or an LLM. Queries are paraphrased away from the clause wording.
The benchmark reports retrieval metrics, paired improvements, bootstrap 95% CIs,
and latency. It is intentionally honest: synthetic results are labeled synthetic.
"""
from __future__ import annotations
import json, math, random, statistics, time
from dataclasses import dataclass
from pathlib import Path
from clauseiq.documents import Clause
from clauseiq.retrieval import HybridRetriever
from clauseiq.reranking import rerank

@dataclass(frozen=True)
class Case:
    query: str
    relevant: str
    split: str

TOPICS = [
("liability", "Supplier has unlimited liability for direct damages.", ["uncapped exposure", "no ceiling on direct loss", "supplier bears unlimited direct liability"]),
("indemnity", "Supplier shall defend and indemnify Customer against third-party claims.", ["vendor covers outside-party claims", "third party claim protection", "supplier indemnification obligation"]),
("termination", "Customer may terminate for convenience on thirty days written notice.", ["end the deal without cause", "termination at will", "cancel for convenience"]),
("renewal", "The agreement automatically renews for successive one-year terms unless notice is given.", ["automatic extension", "rollover into another year", "renewal happens unless notice"]),
("payment", "Overdue invoices accrue interest at 1.5 percent per month until paid.", ["late invoice interest", "finance charge on overdue bills", "interest for delayed payment"]),
("breach notice", "Supplier must notify Customer of a personal data breach within twenty-four hours.", ["privacy incident notification deadline", "report a data breach within a day", "personal information incident notice"]),
("confidentiality", "Confidentiality obligations survive termination for five years.", ["confidentiality continues after expiry", "post termination secrecy period", "survival of nondisclosure duties"]),
("jurisdiction", "The agreement is governed by Delaware law and subject to exclusive jurisdiction there.", ["choice of law and forum", "exclusive court jurisdiction", "Delaware governing law"]),
("uptime", "Supplier guarantees 99.9 percent monthly service availability.", ["service availability commitment", "uptime guarantee", "monthly SLA availability"]),
("ip", "Customer owns intellectual property in deliverables created and paid for by Customer.", ["ownership of work product", "customer owns deliverable IP", "intellectual property rights in outputs"]),
("audit", "Customer may audit relevant Supplier records once each year for compliance.", ["annual compliance audit", "right to inspect vendor records", "customer audit rights"]),
("assignment", "Neither party may assign the agreement without prior written consent.", ["transfer requires approval", "assignment restriction", "consent before contract transfer"]),
("insurance", "Supplier shall maintain commercial general liability insurance of at least two million dollars.", ["minimum vendor insurance", "required liability coverage", "insurance limit"]),
("security", "Supplier shall maintain reasonable administrative, technical, and physical safeguards.", ["information security controls", "security safeguards requirement", "technical and physical protections"]),
("subprocessors", "Supplier must obtain approval before engaging a new subprocessor to process personal data.", ["approval for data subprocessors", "third party processor consent", "vendor cannot add subprocessors freely"]),
("deletion", "Supplier shall delete Customer data within thirty days after termination, subject to legal retention.", ["post termination data deletion", "remove customer information after exit", "data erasure deadline"]),
("force majeure", "Neither party is liable for delay caused by events beyond its reasonable control.", ["excuse for uncontrollable delay", "force majeure protection", "no liability for extraordinary events"]),
("warranty", "Supplier warrants that services will materially conform to the agreed specifications.", ["service conformity promise", "performance warranty", "services match specifications"]),
("limitation", "Neither party is liable for consequential or indirect damages except for specified carve-outs.", ["indirect damages exclusion", "consequential loss waiver", "carveout to damages limitation"]),
("notice", "Formal notices must be sent to the addresses designated in the agreement.", ["contract notice procedure", "where formal notices go", "designated notice address"]),
("records", "Supplier shall retain compliance records for seven years after creation.", ["record retention period", "keep compliance documents", "seven year retention"]),
("dispute", "The parties shall first attempt good-faith negotiation before commencing litigation.", ["negotiation before lawsuit", "pre-litigation dispute process", "good faith dispute resolution"]),
("sla credit", "Customer is entitled to a service credit when monthly availability falls below the SLA target.", ["uptime service credit", "credit for missed availability", "SLA remedy for downtime"]),
("change", "Material changes to the services require written agreement signed by both parties.", ["approval of major service changes", "written change authorization", "both parties sign material modifications"]),
]

DISTRACTORS = [
"The parties will cooperate in good faith and designate operational contacts.",
"Invoices shall contain the purchase order number and billing address.",
"This agreement constitutes the entire understanding between the parties.",
"Headings are for convenience and do not affect interpretation.",
"Each party represents that it has authority to enter into this agreement.",
"Counterparts and electronic signatures are permitted.",
"A waiver must be in writing and signed by the waiving party.",
"If a provision is invalid, the remaining provisions remain effective.",
]

def build_dataset(seed: int = 20260912):
    rng = random.Random(seed)
    clauses=[]; cases=[]
    for idx,(topic,text,queries) in enumerate(TOPICS,1):
        cid=f"c{idx:02d}"; clauses.append(Clause(cid,str(idx),f"{idx}. {topic.upper()}. {text}",0,0))
        for q in queries:
            cases.append(Case(q,cid,"test" if rng.random() < .8 else "holdout"))
        # create near-neighbor clauses to make retrieval non-trivial
        near = f"{idx+100}. {topic.upper()} NOTICE. The parties may discuss {topic} details during routine operations."
        clauses.append(Clause(f"n{idx:02d}",str(idx+100),near,0,0))
    for i,text in enumerate(DISTRACTORS,200): clauses.append(Clause(f"d{i}",str(i),text,0,0))
    return clauses,cases

def metrics(ranked, relevant, k=5):
    ranks=[i+1 for i,x in enumerate(ranked) if x==relevant]
    rr=1/ranks[0] if ranks else 0.0
    r1=float(bool(ranks and ranks[0]<=1)); r3=float(bool(ranks and ranks[0]<=3)); r5=float(bool(ranks and ranks[0]<=5))
    dcg=1/math.log2(ranks[0]+1) if ranks and ranks[0]<=k else 0
    return r1,r3,r5,rr,dcg

def bootstrap_ci(values, seed=7, n=2000):
    rng=random.Random(seed); means=[]; m=len(values)
    for _ in range(n): means.append(statistics.mean(rng.choice(values) for _ in range(m)))
    means.sort(); return means[int(.025*n)], means[int(.975*n)]

def run():
    clauses,cases=build_dataset(); retr=HybridRetriever(clauses)
    methods={
        "tfidf": lambda q: sorted(range(len(clauses)), key=lambda i: retr.vectorizer.transform([q]).dot(retr.X.T).toarray()[0][i], reverse=True),
        "bm25": lambda q: sorted(range(len(clauses)), key=lambda i: retr._bm25(q)[i], reverse=True),
        "hybrid": lambda q: [next(i for i,c in enumerate(clauses) if c.id==h.clause_id) for h in retr.search(q,len(clauses))],
        "rrf": lambda q: [next(i for i,c in enumerate(clauses) if c.id==h.clause_id) for h in retr.search_rrf(q,len(clauses))],
    }
    output={"dataset":{"clauses":len(clauses),"queries":len(cases),"topics":len(TOPICS),"holdout_queries":sum(c.split=='holdout' for c in cases),"seed":20260912},"methods":{}}
    for name,fn in methods.items():
        rows=[]; lat=[]
        for case in cases:
            t=time.perf_counter(); ids=fn(case.query); lat.append((time.perf_counter()-t)*1000)
            r1,r3,r5,rr,nd=metrics([clauses[i].id for i in ids],case.relevant)
            rows.append((case.split,r1,r3,r5,rr,nd))
        def agg(rows):
            return {"recall@1":statistics.mean(r[1] for r in rows),"recall@3":statistics.mean(r[2] for r in rows),"recall@5":statistics.mean(r[3] for r in rows),"mrr":statistics.mean(r[4] for r in rows),"ndcg@5":statistics.mean(r[5] for r in rows)}
        test=[r for r in rows if r[0]=='test']; hold=[r for r in rows if r[0]=='holdout']
        output["methods"][name]={"all":agg(rows),"holdout":agg(hold),"p50_ms":statistics.median(lat),"p95_ms":sorted(lat)[max(0,int(.95*len(lat))-1)]}
    # deterministic reranking stage on full hybrid candidates
    rows=[]
    for case in cases:
        hits=retr.search(case.query,len(clauses)); rr=rerank(case.query,hits,clauses,top_k=len(hits))
        vals=metrics([h.clause_id for h in rr],case.relevant); rows.append((case.split,*vals))
    output["methods"]["hybrid_plus_rerank"]={"all":{k:statistics.mean(r[i] for r in rows) for i,k in enumerate(["recall@1","recall@3","recall@5","mrr","ndcg@5"],1)},"holdout":{k:statistics.mean(r[i] for r in rows if r[0]=='holdout') for i,k in enumerate(["recall@1","recall@3","recall@5","mrr","ndcg@5"],1)}}
    base=[]; improved=[]
    for case in cases:
        # paired MRR delta, hybrid vs RRF
        h=[clauses[i].id for i in methods['hybrid'](case.query)]; r=[clauses[i].id for i in methods['rrf'](case.query)]
        _,_,_,hm,_=metrics(h,case.relevant); _,_,_,rm,_=metrics(r,case.relevant); base.append(hm); improved.append(rm-hm)
    output["paired_rrf_vs_hybrid_mrr_delta"]={"mean":statistics.mean(improved),"bootstrap95":bootstrap_ci(improved)}
    return output

if __name__=='__main__':
    out=run(); print(json.dumps(out,indent=2)); Path(__file__).with_name('benchmark_v2_results.json').write_text(json.dumps(out,indent=2)+'\n')
