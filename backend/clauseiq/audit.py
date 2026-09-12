from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from .database import SessionLocal, AuditEventORM


def record(contract_id: str, event: str, payload: dict | None = None) -> str:
    event_id = uuid4().hex
    with SessionLocal() as s:
        s.add(AuditEventORM(id=event_id, contract_id=contract_id, event=event,
                            payload_json=payload or {}, created_at=datetime.now(timezone.utc).isoformat()))
        s.commit()
    return event_id


def list_events(contract_id: str):
    with SessionLocal() as s:
        return s.query(AuditEventORM).filter_by(contract_id=contract_id).order_by(AuditEventORM.created_at.asc()).all()
