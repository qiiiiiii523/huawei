"""Continue P, reduce P/Q/A/B validation, and save recovery state every500."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reschedule_b7_validation import migrate, without_schedule


def main():
    from baselines.B7.train import validate_settings
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    schedules={'b7_public10.json':2000,'b7_public40.json':1000,'b7_huawei40.json':500}
    # Reject accidental batch/LR/budget/model changes in any downstream stage.
    import subprocess
    for filename,every in schedules.items():
        config=json.loads((ROOT/'configs'/filename).read_text())
        validate_settings(config)
        baseline=json.loads(subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',
                  'show',f'032fc7d:configs/{filename}'],cwd=ROOT))
        if (without_schedule(config)!=without_schedule(baseline)
            or config['training']['validate_every']!=every or config['training'].get('checkpoint_every')!=500):
            raise ValueError(f'Unexpected stage configuration: {filename}')
    migrate(args.source,args.output,args.manifest,ROOT/'configs/b7_public10.json',
            2000,checkpoint_every=500,allow_code_upgrade=True)


if __name__=='__main__':
    main()
