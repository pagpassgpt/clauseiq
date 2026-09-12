from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class RiskLevel(str, Enum): LOW="low"; MEDIUM="medium"; HIGH="high"; CRITICAL="critical"
class FindingCreate(BaseModel):
    clause_id: str; category: str; title: str; rationale: str; quote: str
    risk_level: Optional[RiskLevel]=None
class FindingOut(FindingCreate):
    id: str; severity: RiskLevel; start_offset: int; end_offset: int; grounding_status: str
class ContractOut(BaseModel): id: str; filename: str; content_type: str; clause_count: int
class ReviewOut(BaseModel): review_id: str; contract_id: str; risk_score: float; risk_level: RiskLevel; findings: list[FindingOut]
