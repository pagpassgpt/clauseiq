from clauseiq.risk import score

def test_risk_is_deterministic():
 s,l=score([{"severity":"critical"},{"severity":"high"}]); assert s==60 and l.value=="high"
