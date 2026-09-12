from datetime import datetime, timezone
from uuid import uuid4
from .database import SessionLocal, ContractORM, ClauseORM, FindingORM, ReviewORM
from .documents import parse_bytes, normalize, chunk_clauses
from .retrieval import HybridRetriever
from .reranking import rerank
from .policies import Policy
from .grounding import verify
from .risk import score
from .llm import MockLLM
from .audit import record

class ContractService:
    def ingest(self,filename,data):
        raw,ct=parse_bytes(filename,data); text=normalize(raw); clauses=chunk_clauses(text); cid=uuid4().hex
        with SessionLocal() as s:
            s.add(ContractORM(id=cid,filename=filename,content_type=ct,text=text,created_at=datetime.now(timezone.utc).isoformat()))
            for c in clauses:s.add(ClauseORM(id=f"{cid}:{c.id}",contract_id=cid,section=c.section,text=c.text,start_offset=c.start,end_offset=c.end))
            s.commit()
        record(cid,"contract.ingested",{"filename":filename,"clause_count":len(clauses)})
        return cid,len(clauses)
    def _clauses(self,s,cid):
        rows=s.query(ClauseORM).filter_by(contract_id=cid).order_by(ClauseORM.start_offset).all()
        from .documents import Clause
        return [Clause(r.id,r.section,r.text,r.start_offset,r.end_offset) for r in rows]
    def review(self,cid):
        with SessionLocal() as s:
            clauses=self._clauses(s,cid)
            if not clauses: raise ValueError("Contract not found or has no clauses")
            policy=Policy.default(); retr=HybridRetriever(clauses); hits=retr.search("liability indemnity termination renewal payment",8); hits=rerank("contract risk",hits,clauses,5)
            selected={h.clause_id for h in hits}; candidates=[c for c in clauses if c.id in selected]
            # Deterministic mock is the safe baseline; a real LLM can be plugged in at the boundary.
            proposed=MockLLM().propose(candidates,policy); verified=[]
            for f in proposed:
                c=next((x for x in clauses if x.id==f["clause_id"]),None)
                if not c: continue
                g=verify(c,f["quote"])
                if g.status=="UNSUPPORTED": continue
                fid=uuid4().hex; base=f.copy(); base["clause_id"]=c.id
                s.add(FindingORM(id=fid,contract_id=cid,clause_id=c.id,category=base["category"],severity=base["severity"],title=base["title"],rationale=base["rationale"],quote=base["quote"],start_offset=g.start,end_offset=g.end,grounding_status=g.status,metadata_json={"retrieval":"hybrid"}))
                verified.append((fid,base,g))
            scorev,level=score([{"severity":x[1]["severity"]} for x in verified]); rid=uuid4().hex
            s.add(ReviewORM(id=rid,contract_id=cid,risk_score=scorev,risk_level=level.value,finding_count=len(verified))); s.commit()
            record(cid,"review.completed",{"review_id":rid,"risk_score":scorev,"finding_count":len(verified)})
            return rid,scorev,level,verified
    def findings(self,cid):
        with SessionLocal() as s:
            return s.query(FindingORM).filter_by(contract_id=cid).order_by(FindingORM.severity.desc()).all()

    def compare(self, left_id: str, right_id: str):
        """Deterministic contract comparison at the clause/category level."""
        with SessionLocal() as s:
            left=self._clauses(s,left_id); right=self._clauses(s,right_id)
            if not left or not right: raise ValueError("Both contracts must exist and contain clauses")
            def key(c):
                text=c.text.lower()
                for k,terms in {"liability":["liability","damages","indemn"],"termination":["termination","terminate"],"payment":["payment","invoice","fee"],"confidentiality":["confidential"],"privacy":["personal data","privacy","subprocessor"],"insurance":["insurance","insured"],"renewal":["renew","renewal"]}.items():
                    if any(t in text for t in terms): return k
                return "other"
            L={key(c):c for c in left}; R={key(c):c for c in right}; cats=sorted(set(L)|set(R))
            rows=[]
            for cat in cats:
                a,b=L.get(cat),R.get(cat)
                rows.append({"category":cat,"left_clause_id":a.id if a else None,"right_clause_id":b.id if b else None,"status":"both" if a and b else "left_only" if a else "right_only"})
            return rows
