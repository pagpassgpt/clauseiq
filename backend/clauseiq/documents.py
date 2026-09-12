import re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Clause:
    id: str
    section: str
    text: str
    start: int
    end: int

def parse_bytes(filename: str, data: bytes) -> tuple[str, str]:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace"), "text/plain"
    if ext == ".pdf":
        from pypdf import PdfReader
        import io
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages), "application/pdf"
    if ext == ".docx":
        from docx import Document
        import io
        d = Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    raise ValueError("Unsupported file type; use PDF, DOCX, TXT, or MD")

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()

def chunk_clauses(text: str) -> list[Clause]:
    pat = re.compile(r"(?m)^(?P<section>(?:\d+(?:\.\d+)*|[A-Z])(?:[.)])?)\s+(?P<title>[^\n]{2,100})")
    matches = list(pat.finditer(text))
    if not matches:
        return [Clause("preamble", "PREAMBLE", text, 0, len(text))] if text else []
    out = []
    if matches[0].start() > 0 and text[:matches[0].start()].strip():
        prefix = text[:matches[0].start()]
        out.append(Clause("preamble", "PREAMBLE", prefix.strip(), 0, len(prefix)))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[start:end]
        body = raw.strip()
        left_trim = len(raw) - len(raw.lstrip())
        body_start = start + left_trim
        sec = m.group("section").rstrip(".)")
        out.append(Clause(sec or f"clause-{i+1}", sec, body, body_start, body_start + len(body)))
    return out
