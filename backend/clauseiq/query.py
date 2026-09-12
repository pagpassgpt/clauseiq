"""Conservative legal-domain query expansion.

Expansion is additive: the original query is always retained. The lexicon is
small, explicit, versioned, and benchmarkable rather than an opaque LLM rewrite.
"""
from __future__ import annotations
import re

PHRASE_SYNONYMS = {
    "end the deal": ["terminate", "termination", "cancel"],
    "privacy incident": ["data breach", "security incident", "personal data breach"],
    "notification deadline": ["notice", "notify", "notification period"],
    "data erasure": ["delete data", "deletion", "retention"],
    "confidentiality continues": ["confidentiality survives", "survival"],
    "uptime promise": ["availability", "service level", "sla"],
    "credit for downtime": ["service credit", "sla credit", "availability remedy"],
    "vendor coverage": ["insurance", "liability insurance", "coverage"],
    "uncapped direct exposure": ["uncapped liability", "unlimited liability", "no cap"],
    "data processors": ["subprocessor", "processor", "data processor"],
    "data may be stored": ["data residency", "geographic regions", "hosting location"],
    "hiring restriction": ["non solicitation", "solicit employees", "employee poaching"],
    "source code backup": ["escrow", "source code escrow"],
    "release mechanism": ["release trigger", "trigger events", "escrow release"],
    "pricing adjustment": ["price increase", "fee increase", "pricing"],
    "late invoice": ["overdue invoice", "payment", "interest"],
}

LEGAL_SYNONYMS = {
    "uncapped": ["unlimited", "without limitation", "no cap"],
    "liability": ["damages", "losses", "exposure"],
    "indemnity": ["indemnify", "hold harmless", "defend"],
    "terminate": ["termination", "cancel", "end", "exit"],
    "renewal": ["renew", "extension", "rollover", "auto-renew"],
    "payment": ["invoice", "fee", "interest", "late"],
    "breach": ["incident", "security incident", "data breach"],
    "confidentiality": ["confidential", "nondisclosure", "secrecy"],
    "jurisdiction": ["governing law", "forum", "courts", "venue"],
    "uptime": ["availability", "service level", "sla"],
    "ownership": ["owns", "intellectual property", "ip", "rights"],
    "audit": ["inspect", "records", "compliance review"],
    "assignment": ["assign", "transfer", "consent"],
    "insurance": ["coverage", "policy", "insured", "liability insurance"],
    "security": ["safeguards", "controls", "protection", "security controls"],
    "subprocessor": ["processor", "third party processor", "data processor"],
    "deletion": ["delete", "erase", "erasure", "remove", "retention"],
    "warranty": ["warrants", "conform", "specifications", "performance"],
    "notice": ["notify", "notification", "written notice"],
    "records": ["retain", "retention", "documents", "files"],
    "dispute": ["negotiation", "litigation", "resolution", "court"],
    "credit": ["remedy", "compensation", "service credit"],
    "change": ["modification", "amendment", "approval"],
    "residency": ["geographic", "region", "hosting", "storage"],
    "remedy": ["fix", "correct", "remediate", "cure"],
    "solicitation": ["solicit", "poaching", "hiring"],
    "price": ["fee", "fees", "pricing", "increase"],
    "export": ["trade", "export control", "sanctions"],
    "escrow": ["source code", "release", "backup"],
}

def expand_query(query: str, max_terms: int = 28) -> str:
    lowered = query.lower()
    extra = []
    # Phrase expansion catches realistic multi-intent wording before token-level expansion.
    for phrase, synonyms in PHRASE_SYNONYMS.items():
        if phrase in lowered:
            extra.extend(synonyms)
    terms = re.findall(r"(?u)\b\w[\w-]*\b", lowered)
    for term in terms:
        extra.extend(LEGAL_SYNONYMS.get(term, ()))
    out = list(dict.fromkeys(terms + extra))[:max_terms]
    return " ".join(out)

def split_multi_hop(query: str) -> list[str]:
    """Split common multi-intent questions without an LLM."""
    parts = re.split(r"\s+(?:and|also|plus|;|,)\s+", query, flags=re.I)
    return [p.strip() for p in parts if len(p.strip()) >= 3]
