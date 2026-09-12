from dataclasses import dataclass
@dataclass(frozen=True)
class Grounded:
    status:str; start:int; end:int; quote:str

def verify(clause, quote):
    if not quote or not quote.strip(): return Grounded("UNSUPPORTED",-1,-1,quote)
    q=quote.strip(); idx=clause.text.find(q)
    if idx>=0: return Grounded("EXACT",clause.start+idx,clause.start+idx+len(q),q)
    # whitespace-normalized containment is PARTIAL, never silently upgraded to exact.
    norm=lambda s:" ".join(s.split())
    nq=norm(q); nt=norm(clause.text)
    if nq and nq in nt: return Grounded("PARTIAL",clause.start,clause.end,q)
    return Grounded("UNSUPPORTED",-1,-1,q)
