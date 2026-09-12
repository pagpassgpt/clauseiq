from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .database import init_db, SessionLocal, ContractORM, ReviewORM
from .audit import list_events
from .service import ContractService
from .config import settings
app=FastAPI(title="ClauseIQ",version="6.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://localhost:3000"],allow_methods=["*"],allow_headers=["*"])
@app.on_event("startup")
def startup(): init_db()
@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/contracts")
async def upload_contract(file:UploadFile=File(...)):
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB limit")
    try:
        cid,n=ContractService().ingest(file.filename or "upload",data)
        return {"id":cid,"clause_count":n}
    except ValueError as e: raise HTTPException(400,str(e))
@app.post("/contracts/{contract_id}/review")
def review(contract_id:str):
    try:
        rid,rs,level,verified=ContractService().review(contract_id)
        return {"review_id":rid,"contract_id":contract_id,"risk_score":rs,"risk_level":level.value,"findings":[{"id":fid,"clause_id":f["clause_id"],"category":f["category"],"title":f["title"],"rationale":f["rationale"],"quote":f["quote"],"severity":f["severity"],"start_offset":g.start,"end_offset":g.end,"grounding_status":g.status} for fid,f,g in verified]}
    except ValueError as e: raise HTTPException(404,str(e))
@app.get("/contracts/{contract_id}/findings")
def findings(contract_id:str):
    return [{"id":x.id,"clause_id":x.clause_id,"category":x.category,"severity":x.severity,"title":x.title,"rationale":x.rationale,"quote":x.quote,"start_offset":x.start_offset,"end_offset":x.end_offset,"grounding_status":x.grounding_status} for x in ContractService().findings(contract_id)]
@app.get("/contracts")
def contracts():
    with SessionLocal() as s:return [{"id":x.id,"filename":x.filename,"content_type":x.content_type} for x in s.query(ContractORM).order_by(ContractORM.created_at.desc()).all()]


@app.get("/contracts/{contract_id}/audit")
def audit(contract_id: str):
    try:
        return [{"id":e.id,"event":e.event,"payload":e.payload_json,"created_at":e.created_at} for e in list_events(contract_id)]
    except Exception as e:
        raise HTTPException(404,str(e))

@app.get("/contracts/{contract_id}/export")
def export_review(contract_id: str):
    try:
        rows=ContractService().findings(contract_id)
        return {"contract_id":contract_id,"findings":[{"id":x.id,"clause_id":x.clause_id,"category":x.category,"severity":x.severity,"title":x.title,"rationale":x.rationale,"quote":x.quote,"start_offset":x.start_offset,"end_offset":x.end_offset,"grounding_status":x.grounding_status} for x in rows]}
    except Exception as e:
        raise HTTPException(404,str(e))

@app.get("/contracts/{left_id}/compare/{right_id}")
def compare_contracts(left_id: str, right_id: str):
    try: return {"left_contract_id":left_id,"right_contract_id":right_id,"differences":ContractService().compare(left_id,right_id)}
    except ValueError as e: raise HTTPException(404,str(e))
