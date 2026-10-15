"""Explicit paired FM continuation; no training on import."""
import argparse
from .runtime import run
from . import GROUPS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared',required=True)
    parser.add_argument('--group',choices=GROUPS,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--execute-training',action='store_true')
    args=parser.parse_args()
    if not args.execute_training: parser.error('Training requires explicit --execute-training')
    run(args)


if __name__=='__main__': main()
