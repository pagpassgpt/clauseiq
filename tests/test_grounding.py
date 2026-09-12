from clauseiq.documents import Clause
from clauseiq.grounding import verify

def test_grounding_exact_and_rejects_fake_quote():
 c=Clause("1","1","Supplier has unlimited liability.",10,42)
 assert verify(c,"unlimited liability").status=="EXACT"
 assert verify(c,"Supplier has unlimited indemnity").status=="UNSUPPORTED"
