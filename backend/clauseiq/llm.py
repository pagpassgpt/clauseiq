import json, httpx
class MockLLM:
    def propose(self, clauses, policy):
        findings=[]
        for c in clauses:
            for r in policy.assess(c.text):
                findings.append({"clause_id":c.id,"category":r.category,"title":r.description,"rationale":r.description,"quote":next((t for t in r.terms if t.lower() in c.text.lower()),c.text[:160]),"severity":r.level})
        return findings
class OpenAICompatibleLLM:
    def __init__(self,base_url,api_key,model,temp=1.0,top_p=.95): self.base=base_url.rstrip('/'); self.key=api_key; self.model=model; self.temp=temp; self.top_p=top_p
    def chat(self,messages,tools=None):
        payload={"model":self.model,"messages":messages,"temperature":self.temp,"top_p":self.top_p}
        if tools: payload["tools"]=tools
        r=httpx.post(self.base+"/chat/completions",headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"},json=payload,timeout=90); r.raise_for_status(); return r.json()
