"""Optional OpenAI-compatible model benchmark for ClauseIQ's bounded agent.

This script does not require a model key for CI. Set CLAUSEIQ_LLM_BASE_URL,
CLAUSEIQ_LLM_API_KEY and CLAUSEIQ_LLM_MODEL (or the LLM_* aliases in .env) to
run it against NVIDIA/OpenRouter or another compatible endpoint.
"""
from __future__ import annotations
import json, os, statistics, time
from clauseiq.agent import LLMReviewAgent
from clauseiq.documents import Clause
from clauseiq.llm import OpenAICompatibleLLM

CASES=[
 ("Find the uncapped liability provision and cite the exact evidence.", {"c1"}),
 ("Find the automatic renewal provision and cite the exact evidence.", {"c2"}),
 ("Find the termination for convenience provision and cite the exact evidence.", {"c3"}),
 ("Find both liability and indemnity provisions.", {"c1","c4"}),
]

def main():
 base=os.getenv("CLAUSEIQ_LLM_BASE_URL",os.getenv("LLM_BASE_URL","")); key=os.getenv("CLAUSEIQ_LLM_API_KEY",os.getenv("LLM_API_KEY","")); model=os.getenv("CLAUSEIQ_LLM_MODEL",os.getenv("LLM_MODEL",""))
 if not (base and key and model):
  print(json.dumps({"status":"skipped","reason":"set CLAUSEIQ_LLM_BASE_URL, CLAUSEIQ_LLM_API_KEY and CLAUSEIQ_LLM_MODEL"},indent=2)); return
 clauses=[Clause("c1","1","LIMITATION: Supplier has unlimited liability for direct damages.",0,0),Clause("c2","2","The agreement automatically renews for one-year terms.",0,0),Clause("c3","3","Customer may terminate for convenience on thirty days notice.",0,0),Clause("c4","4","Supplier shall indemnify Customer against third-party claims.",0,0)]
 agent=LLMReviewAgent(clauses); llm=OpenAICompatibleLLM(base,key,model)
 rows=[]
 for task,gold in CASES:
  t=time.perf_counter(); out=agent.run_llm(task,llm); ms=(time.perf_counter()-t)*1000
  found={c.result.get("clause_id") for c in out["trajectory"].calls if c.name=="record_finding" and c.ok and isinstance(c.result,dict)}
  rows.append({"task":task,"gold":sorted(gold),"found":sorted(x for x in found if x),"gold_recall":len(found&gold)/len(gold),"failed_calls":sum(not c.ok for c in out["trajectory"].calls),"tool_calls":len(out["trajectory"].calls),"latency_ms":ms,"status":out["trajectory"].status})
 result={"model":model,"cases":len(rows),"mean_gold_recall":statistics.mean(x["gold_recall"] for x in rows),"failure_rate":statistics.mean(x["failed_calls"]>0 for x in rows),"mean_tool_calls":statistics.mean(x["tool_calls"] for x in rows),"p50_latency_ms":statistics.median(x["latency_ms"] for x in rows),"rows":rows}
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
