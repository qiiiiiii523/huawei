"""Call the unchanged organizer reference evaluator and save its full JSON report."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

from competition_interface import output
from baselines.B5_best.common import write


def evaluator():
    # The supplied evaluate.py uses `from output import ...` as a standalone
    # sibling import. Bind that exact official module while importing it.
    old=sys.modules.get('output')
    try:
        sys.modules['output']=output
        return importlib.import_module('competition_interface.evaluate')
    finally:
        if old is None: sys.modules.pop('output',None)
        else: sys.modules['output']=old


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred-dir',required=True);p.add_argument('--gold-dir',required=True)
    p.add_argument('--report',required=True)
    args=p.parse_args();path=Path(args.report)
    if path.exists(): raise FileExistsError('Keep old reference reports; choose a new filename')
    report=evaluator().evaluate_directory(args.pred_dir,args.gold_dir)
    write(path,report)
    print(report['summary'],flush=True)


if __name__=='__main__': main()
