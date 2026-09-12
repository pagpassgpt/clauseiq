from __future__ import annotations
import json, random
from pathlib import Path
from clauseiq.grounding import verify
from clauseiq.risk import score

def run(n=1000,seed=20260912):
    exact=rejected=deterministic=0
    text="LIMITATION OF LIABILITY: Neither party shall be liable for consequential damages."
    class C: pass
    c=C(); c.text=text; c.start=0; c.end=len(text)
    for _ in range(n):
        exact += verify(c,"consequential damages").status=="EXACT"
        rejected += verify(c,"consequential moon damages").status=="UNSUPPORTED"
        findings=[{"severity":"high","category":"liability"},{"severity":"medium","category":"limitation"}]
        deterministic += score(findings)==score(findings)
    return {"samples":n,"exact_grounding_rate":exact/n,"unsupported_rejection_rate":rejected/n,"risk_determinism_rate":deterministic/n}
if __name__=='__main__':
    out=run(); Path(__file__).with_name('quality_v3_results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
