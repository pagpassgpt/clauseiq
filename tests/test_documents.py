from clauseiq.documents import normalize,chunk_clauses

def test_offsets_are_source_spans():
 t=normalize("1. Liability\nSupplier has unlimited liability.\n2. Payment\nNet 30 days.")
 cs=chunk_clauses(t); assert [c.id for c in cs]==["1","2"]
 for c in cs: assert t[c.start:c.end].strip()==c.text
