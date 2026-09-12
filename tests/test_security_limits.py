import pytest
from clauseiq.documents import parse_bytes

def test_filename_path_is_not_executed_or_opened():
    text, kind = parse_bytes("../../untrusted.txt", b"safe")
    assert text == "safe" and kind == "text/plain"

def test_unsupported_extension_rejected():
    with pytest.raises(ValueError):
        parse_bytes("payload.exe", b"not executable")
