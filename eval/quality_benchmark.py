"""High-volume deterministic safety gates for ClauseIQ."""
import json
from pathlib import Path
from clauseiq.documents import Clause
from clauseiq.grounding import verify
from clauseiq.policies import Policy
from clauseiq.risk import score

def main():
    c=Clause('7.2','7.2','Supplier shall have unlimited liability for direct damages.',100,159)
    exact=sum(verify(c,'unlimited liability for direct damages.').status=='EXACT' for _ in range(500))
    rejected=sum(verify(c,f'unlimited liability for direct damages {i}').status=='UNSUPPORTED' for i in range(500))
    results=[score([{'severity':'critical'},{'severity':'high'}]) for _ in range(500)]
    policy=Policy.default(); policy_hits=len(policy.assess(c.text))
    result={'iterations':500,'grounding_exact_rate':exact/500,'unsupported_evidence_rejection_rate':rejected/500,'risk_determinism_rate':float(len(set(map(str,results)))==1),'policy_match_count':policy_hits,'all_gates_pass':exact==500 and rejected==500 and len(set(map(str,results)))==1}
    print(json.dumps(result,indent=2)); Path('eval/quality_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
