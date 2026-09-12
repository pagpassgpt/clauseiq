from dataclasses import dataclass
from pathlib import Path
import yaml
@dataclass(frozen=True)
class Rule:
    category:str; level:str; weight:float; terms:tuple[str,...]; description:str
class Policy:
    def __init__(self,rules): self.rules=rules
    @classmethod
    def default(cls):
        return cls([
          Rule("liability","critical",35,("unlimited liability","uncapped liability","without limitation"),"Uncapped liability exposure"),
          Rule("indemnity","high",25,("indemnify","hold harmless"),"Broad indemnification"),
          Rule("termination","high",20,("terminate for convenience","termination for convenience"),"Asymmetric convenience termination"),
          Rule("renewal","medium",10,("automatically renew","auto-renew"),"Automatic renewal"),
          Rule("payment","medium",10,("late fee","interest at"),"Payment penalty"),
        ])
    @classmethod
    def from_yaml(cls,path):
        d=yaml.safe_load(Path(path).read_text()); return cls([Rule(x["category"],x["level"],float(x["weight"]),tuple(x.get("terms",[])),x.get("description","")) for x in d["rules"]])
    def assess(self,text):
        low=text.lower(); hits=[]
        for r in self.rules:
            if any(t.lower() in low for t in r.terms): hits.append(r)
        return hits
