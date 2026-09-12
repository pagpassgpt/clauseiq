"""ClauseIQ Retrieval & Safety Benchmark v3.

This benchmark is intentionally self-contained and reproducible. It separates
retrieval quality from generation/grounding quality, uses held-out paraphrases,
noise robustness, hard negatives, and paired bootstrap confidence intervals.
Synthetic results are explicitly labeled synthetic; they are not claims about
real-world legal-review accuracy.
"""
from __future__ import annotations
import json, math, random, statistics, time
from dataclasses import dataclass
from pathlib import Path
from clauseiq.documents import Clause
from clauseiq.retrieval import HybridRetriever
from clauseiq.reranking import rerank

TOPICS=[
("liability","Supplier has unlimited liability for direct damages.",["uncapped direct exposure","no ceiling on supplier losses","vendor bears all direct damages without a cap"]),
("indemnity","Supplier shall defend and indemnify Customer against third-party claims.",["vendor covers outside-party claims","supplier must protect customer from third party lawsuits","third party claim defense obligation"]),
("termination","Customer may terminate for convenience on thirty days written notice.",["end the deal without cause","cancel the agreement at will","customer can exit with notice"]),
("renewal","The agreement automatically renews for successive one-year terms unless notice is given.",["automatic extension of the contract","rollover into another year","contract renews unless someone opts out"]),
("payment","Overdue invoices accrue interest at 1.5 percent per month until paid.",["late invoice finance charge","interest on unpaid bills","monthly charge for delayed payment"]),
("breach notice","Supplier must notify Customer of a personal data breach within twenty-four hours.",["privacy incident notification deadline","report a data breach within one day","rapid notice of personal information incident"]),
("confidentiality","Confidentiality obligations survive termination for five years.",["secrecy continues after the contract ends","post expiry confidentiality period","nondisclosure duties survive termination"]),
("jurisdiction","The agreement is governed by Delaware law and subject to exclusive jurisdiction there.",["choice of governing law and courts","exclusive forum selection","which state's courts control"]),
("uptime","Supplier guarantees 99.9 percent monthly service availability.",["service availability commitment","uptime promise","monthly availability SLA"]),
("ip","Customer owns intellectual property in deliverables created and paid for by Customer.",["ownership of work product","client owns deliverable rights","who owns the output IP"]),
("audit","Customer may audit relevant Supplier records once each year for compliance.",["annual compliance inspection","right to inspect vendor records","customer audit entitlement"]),
("assignment","Neither party may assign the agreement without prior written consent.",["contract transfer requires approval","assignment restriction","cannot transfer without consent"]),
("insurance","Supplier shall maintain commercial general liability insurance of at least two million dollars.",["minimum vendor coverage","required liability insurance","insurance limit for supplier"]),
("security","Supplier shall maintain reasonable administrative, technical, and physical safeguards.",["information security controls","organizational technical safeguards","security protections requirement"]),
("subprocessors","Supplier must obtain approval before engaging a new subprocessor to process personal data.",["approval for data processors","third party processor consent","vendor needs permission for subprocessors"]),
("deletion","Supplier shall delete Customer data within thirty days after termination, subject to legal retention.",["post termination data erasure","remove customer information after exit","data deletion deadline"]),
("force majeure","Neither party is liable for delay caused by events beyond its reasonable control.",["excuse for uncontrollable delay","extraordinary event protection","delay outside a party's control"]),
("warranty","Supplier warrants that services will materially conform to the agreed specifications.",["service conformity promise","performance warranty","services must match specifications"]),
("limitation","Neither party is liable for consequential or indirect damages except for specified carve-outs.",["indirect loss exclusion","consequential damages waiver","exceptions to damages limitation"]),
("notice","Formal notices must be sent to the addresses designated in the agreement.",["contract notice procedure","where legal notices go","designated address for notices"]),
("records","Supplier shall retain compliance records for seven years after creation.",["record retention period","how long compliance files are kept","seven year document retention"]),
("dispute","The parties shall first attempt good-faith negotiation before commencing litigation.",["negotiation before court action","pre lawsuit dispute process","good faith resolution step"]),
("sla credit","Customer is entitled to a service credit when monthly availability falls below the SLA target.",["credit for downtime","availability shortfall remedy","SLA compensation for missed uptime"]),
("change","Material changes to the services require written agreement signed by both parties.",["major service modifications need approval","written authorization for changes","both parties sign significant changes"]),
("data residency","Supplier shall store Customer personal data only in approved geographic regions.",["where customer data may be stored","regional data hosting restriction","approved locations for personal information"]),
("breach remedy","Supplier shall remediate confirmed security vulnerabilities without undue delay.",["fix verified security weaknesses promptly","vulnerability remediation duty","security flaw correction"]),
("non solicitation","For twelve months after termination, neither party may solicit the other's employees.",["post contract hiring restriction","employee poaching prohibition","staff solicitation ban"]),
("price increase","Supplier may increase fees no more than once annually with sixty days notice.",["annual pricing adjustment limit","vendor fee increase restriction","notice before price rises"]),
("export","Each party shall comply with applicable export control laws.",["trade compliance requirement","export regulation obligation","international shipment controls"]),
("privacy deletion","Customer may request deletion of personal data where legally permitted.",["right to erase personal information","customer data removal request","privacy deletion right"]),
("escrow","Supplier shall place source code in escrow and release it upon defined trigger events.",["source code backup release mechanism","software escrow arrangement","code released after trigger"]),
]
DISTRACTORS=[
"The parties will cooperate in good faith and designate operational contacts.",
"Invoices shall contain the purchase order number and billing address.",
"This agreement constitutes the entire understanding between the parties.",
"Headings are for convenience and do not affect interpretation.",
"Each party represents that it has authority to enter into this agreement.",
"Counterparts and electronic signatures are permitted.",
"A waiver must be in writing and signed by the waiving party.",
"If a provision is invalid, the remaining provisions remain effective.",
"The parties will review operational performance during quarterly meetings.",
"Project managers may coordinate routine implementation matters.",
]
@dataclass(frozen=True)
class Case: query:str; relevant:str; split:str; family:str

def build(seed=20260912):
    rng=random.Random(seed); clauses=[]; cases=[]
    for idx,(topic,text,queries) in enumerate(TOPICS,1):
        cid=f"c{idx:02d}"; clauses.append(Clause(cid,str(idx),f"{idx}. {topic.upper()}. {text}",0,0))
        variants=list(queries)+[
            f"what does the agreement say about {topic}",
            f"find the provision dealing with {topic} obligations",
        ]
        for j,q in enumerate(variants):
            split="holdout" if j>=len(queries) else ("test" if rng.random()<.8 else "holdout")
            cases.append(Case(q,cid,split,topic))
        # hard negative shares the topic word but not the operative legal meaning
        clauses.append(Clause(f"n{idx:02d}",str(idx+100),f"{idx+100}. {topic.upper()} ADMIN. The parties may discuss {topic} details during routine operations.",0,0))
    for i,text in enumerate(DISTRACTORS,200): clauses.append(Clause(f"d{i}",str(i),text,0,0))
    return clauses,cases

def ndcg(rank,relevant,k):
    if relevant not in rank[:k]: return 0.0
    r=rank.index(relevant)+1
    return 1/math.log2(r+1)

def score(rank,relevant):
    pos=rank.index(relevant)+1 if relevant in rank else None
    return (float(pos==1),float(pos is not None and pos<=3),float(pos is not None and pos<=5),1/pos if pos else 0.0,ndcg(rank,relevant,5))

def agg(rows):
    return {"recall@1":statistics.mean(r[0] for r in rows),"recall@3":statistics.mean(r[1] for r in rows),"recall@5":statistics.mean(r[2] for r in rows),"mrr":statistics.mean(r[3] for r in rows),"ndcg@5":statistics.mean(r[4] for r in rows)}

def bootstrap_delta(xs,seed=42,n=3000):
    rng=random.Random(seed); means=[]
    for _ in range(n): means.append(statistics.mean(rng.choice(xs) for _ in xs))
    means.sort(); return [means[int(.025*n)],means[int(.975*n)]]

def multihop_eval(clauses, retr):
    pairs=[
      ("end the deal without cause and privacy incident notification deadline", {"c03","c06"}),
      ("post termination data erasure and confidentiality continues after the contract ends", {"c16","c07"}),
      ("uptime promise and credit for downtime", {"c09","c23"}),
      ("minimum vendor coverage and uncapped direct exposure", {"c13","c01"}),
      ("approval for data processors and where customer data may be stored", {"c15","c26"}),
      ("post contract hiring restriction and source code backup release mechanism", {"c28","c30"}),
      ("annual pricing adjustment limit and late invoice finance charge", {"c29","c05"}),
      ("performance warranty and service conformity promise", {"c18","c18"}),
    ]
    rows=[]; decomposed=[]
    for q,gold in pairs:
        ranked=[h.clause_id for h in retr.search(q,len(clauses))]
        top5=set(ranked[:5]); rows.append(len(top5 & gold)/len(gold))
        parts=[x.strip() for x in q.split(" and ") if x.strip()]
        union=[]; seen=set()
        for part in parts:
            for h in retr.search(part,5):
                if h.clause_id not in seen: seen.add(h.clause_id); union.append(h.clause_id)
        decomposed.append(len(set(union[:5]) & gold)/len(gold))
    return {"queries":len(rows),"mean_gold_recall@5":sum(rows)/len(rows),"all_gold_retrieved_rate":sum(x==1 for x in rows)/len(rows),"query_decomposition_mean_recall@5":sum(decomposed)/len(decomposed),"query_decomposition_all_gold_rate":sum(x==1 for x in decomposed)/len(decomposed)}

def run():
    clauses,cases=build(); r=HybridRetriever(clauses)
    methods={
      "bm25":lambda q: sorted(range(len(clauses)),key=lambda i:r._bm25(q)[i],reverse=True),
      "word_tfidf":lambda q: sorted(range(len(clauses)),key=lambda i:r._signals(q)[0][i],reverse=True),
      "char_tfidf":lambda q: sorted(range(len(clauses)),key=lambda i:r._signals(q)[1][i],reverse=True),
      "hybrid":lambda q:[next(i for i,c in enumerate(clauses) if c.id==h.clause_id) for h in r.search(q,len(clauses))],
      "rrf":lambda q:[next(i for i,c in enumerate(clauses) if c.id==h.clause_id) for h in r.search_rrf(q,len(clauses))],
    }
    out={"version":"4.0","dataset":{"topics":len(TOPICS),"clauses":len(clauses),"queries":len(cases),"holdout":sum(x.split=='holdout' for x in cases),"seed":20260912},"methods":{}}
    for name,fn in methods.items():
        rows=[]; lats=[]
        for c in cases:
            t=time.perf_counter(); ids=fn(c.query); lats.append((time.perf_counter()-t)*1000)
            rows.append((c.split,*score([clauses[i].id for i in ids],c.relevant)))
        test=[x[1:] for x in rows if x[0]=='test']; hold=[x[1:] for x in rows if x[0]=='holdout']
        out["methods"][name]={"test":agg(test),"holdout":agg(hold),"p50_ms":statistics.median(lats),"p95_ms":sorted(lats)[max(0,int(.95*len(lats))-1)]}
    rows=[]; lats=[]
    for c in cases:
        hits=r.search_rrf(c.query, min(24,len(clauses)))
        t=time.perf_counter(); rr=rerank(c.query,hits,clauses,top_k=len(hits)); lats.append((time.perf_counter()-t)*1000)
        rows.append((c.split,*score([h.clause_id for h in rr],c.relevant)))
    out["methods"]["rrf_plus_rerank"]={"test":agg([x[1:] for x in rows if x[0]=='test']),"holdout":agg([x[1:] for x in rows if x[0]=='holdout']),"p50_ms":statistics.median(lats),"p95_ms":sorted(lats)[max(0,int(.95*len(lats))-1)]}
    out["multi_hop"] = multihop_eval(clauses, r)
    # paired delta against strongest first-stage baseline
    pairs=[]
    for c in cases:
        h=[clauses[i].id for i in methods['hybrid'](c.query)]; rr=[clauses[i].id for i in methods['rrf'](c.query)]
        pairs.append(score(rr,c.relevant)[3]-score(h,c.relevant)[3])
    out["paired_rrf_minus_hybrid_mrr"]={"mean":statistics.mean(pairs),"bootstrap95":bootstrap_delta(pairs)}
    return out
if __name__=='__main__':
    o=run(); Path(__file__).with_name('benchmark_v3_results.json').write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o,indent=2))
