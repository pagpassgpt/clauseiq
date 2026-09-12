"""Backward-compatible entry point for the external evaluation suite."""
from .external_suite import evaluate
import argparse, json
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--split",default="test"); p.add_argument("--out",default=None); a=p.parse_args()
    result=evaluate(a.dataset,a.split)
    print(json.dumps(result,indent=2))
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
