from pathlib import Path
from clauseiq.documents import normalize,chunk_clauses
from clauseiq.policies import Policy
from clauseiq.llm import MockLLM

def main():
 tp=fp=fn=0
 for p in Path(__file__).parent.joinpath("contracts").glob("*.txt"):
  cs=chunk_clauses(normalize(p.read_text())); pred={c.id for c in cs if Policy.default().assess(c.text)}
  gold={"1"} if p.name=="risky.txt" else set(); tp+=len(pred&gold); fp+=len(pred-gold); fn+=len(gold-pred)
 precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
 print({"precision":round(precision,3),"recall":round(recall,3),"f1":round(f1,3),"tp":tp,"fp":fp,"fn":fn})
if __name__=="__main__": main()
