from clauseiq.agent import LLMReviewAgent
from clauseiq.documents import Clause

class FakeLLM:
    def __init__(self): self.n=0
    def chat(self,messages,tools=None):
        self.n+=1
        if self.n==1:
            return {"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"search_contract","arguments":"{\"query\":\"uncapped liability\",\"top_k\":3}"}}]}}]}
        return {"choices":[{"message":{"role":"assistant","content":"The relevant clause is c1."}}]}

def test_llm_agent_dispatch_is_bounded_and_scoped():
    clauses=[Clause("c1","1","Supplier has unlimited liability for direct damages.",0,50)]
    out=LLMReviewAgent(clauses).run_llm("find liability",FakeLLM(),max_steps=3)
    assert out["trajectory"].status=="completed"
    assert len(out["trajectory"].calls)==1
    assert out["trajectory"].calls[0].ok
