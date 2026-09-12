import os, tempfile
os.environ["DATABASE_URL"]="sqlite:///:memory:"
# Import-time engine uses current env; this test focuses on the service path with the default test DB setup.
from clauseiq.documents import parse_bytes, normalize, chunk_clauses

def test_ingestion_formats():
 text,ct=parse_bytes("x.txt",b"1. Liability\nSupplier has unlimited liability.")
 assert ct=="text/plain" and len(chunk_clauses(normalize(text)))==1
