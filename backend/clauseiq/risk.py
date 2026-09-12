from .models import RiskLevel

def score(findings):
    weights={"critical":35,"high":25,"medium":10,"low":3}
    raw=min(100,sum(weights.get(f["severity"],0) for f in findings))
    level=RiskLevel.CRITICAL if raw>=70 else RiskLevel.HIGH if raw>=40 else RiskLevel.MEDIUM if raw>=15 else RiskLevel.LOW
    return float(raw),level
