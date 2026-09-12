import re
from collections import Counter

def _tok(s): return re.findall(r"(?u)\b\w+\b", s.lower())

def rerank(query, hits, clauses, top_k=5):
    """Deterministic second-stage reranker used in offline/CI mode.

    It combines query-term coverage, rare-term coverage, exact phrase overlap,
    and the first-stage retrieval score. An optional neural CrossEncoder can be
    enabled through CLAUSEIQ_USE_CROSS_ENCODER=1.
    """
    by={c.id:c for c in clauses}; q=_tok(query); qset=set(q)
    try:
        import os
        if os.getenv("CLAUSEIQ_USE_CROSS_ENCODER","0") == "1":
            from sentence_transformers import CrossEncoder
            model=CrossEncoder(os.getenv("CLAUSEIQ_RERANKER","cross-encoder/ms-marco-MiniLM-L-6-v2"))
            pairs=[(query,by[h.clause_id].text) for h in hits]
            scores=model.predict(pairs)
            return [h for _,h in sorted(zip(scores,hits),key=lambda x:float(x[0]),reverse=True)[:top_k]]
    except Exception:
        pass
    scored=[]
    for h in hits:
        text=by[h.clause_id].text.lower(); dt=_tok(text); ds=set(dt)
        coverage=len(qset & ds)/max(1,len(qset))
        rare=sum(1 for t in q if t in ds and len(t)>5)/max(1,len(q))
        phrase=1.0 if query.lower() in text else 0.0
        density=sum(Counter(dt)[t] for t in qset)/max(1,len(dt))
        score=.48*h.score+.27*coverage+.15*rare+.07*phrase+.03*density
        scored.append((score,h))
    return [h for _,h in sorted(scored,key=lambda x:x[0],reverse=True)[:top_k]]
