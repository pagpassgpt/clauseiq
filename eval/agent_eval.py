"""Trajectory-level evaluation: tool selection, grounding, recovery, and efficiency."""
from __future__ import annotations
import json, statistics
from pathlib import Path
from clauseiq.agent import ReviewAgent
from clauseiq.documents import Clause

CASES=[
 ("Find the uncapped liability provision", {"c1"}),
 ("Find the automatic renewal provision", {"c2"}),
 ("Find termination for convenience", {"c3"}),
 ("Find liability and indemnity", {"c1","c4"}),
]

def run():
    clauses=[Clause("c1","1","LIMITATION: Supplier has unlimited liability for direct damages.",0,0),Clause("c2","2","The agreement automatically renews for one-year terms.",0,0),Clause("c3","3","Customer may terminate for convenience on thirty days notice.",0,0),Clause("c4","4","Supplier shall indemnify Customer against third-party claims.",0,0)]
    agent=ReviewAgent(clauses); rows=[]
    for task,gold in CASES:
        tr=agent.run(task); found=set()
        for call in tr.calls:
            if call.name=="search_contract" and call.ok:
                found |= {x["clause_id"] for x in call.result}
        rows.append({"task":task,"gold":sorted(gold),"retrieved":sorted(found),"gold_recall":len(found&gold)/len(gold),"tool_calls":len(tr.calls),"failed_calls":sum(not c.ok for c in tr.calls),"status":tr.status})
    return {"cases":len(rows),"mean_gold_recall":statistics.mean(r["gold_recall"] for r in rows),"failure_rate":statistics.mean(r["failed_calls"]>0 for r in rows),"mean_tool_calls":statistics.mean(r["tool_calls"] for r in rows),"rows":rows}
if __name__=='__main__':
    out=run(); Path(__file__).with_name('agent_eval_results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
