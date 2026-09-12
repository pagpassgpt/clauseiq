from clauseiq.documents import Clause
from clauseiq.retrieval import HybridRetriever

def test_hybrid_keeps_negation_terms():
 cs=[Clause("1","1","Supplier shall not be liable for indirect damages.",0,50),Clause("2","2","Payment is due in 30 days.",51,80)]
 h=HybridRetriever(cs).search("not liable indirect damages",2); assert h[0].clause_id=="1"

def test_rrf_search_is_contract_scoped_and_deterministic():
    cs=[Clause("a","1","Unlimited liability applies to supplier.",0,40),Clause("b","2","Payment is due monthly.",41,70)]
    r=HybridRetriever(cs)
    assert [x.clause_id for x in r.search_rrf("liability",2)] == ["a","b"]
    assert [x.clause_id for x in r.search_rrf("liability",2)] == ["a","b"]
