#!/usr/bin/env bash
set -euo pipefail
gpu=2
repo=/home/qht/huawei
output=
while (($#)); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --repository-root) repo="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical device number' >&2; exit 2; }
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
output=${output:-"$repo/results/B5-slow-public/P_deployment_$(date +%Y%m%d_%H%M%S)_$$"}
[[ ! -e "$output" ]] || { echo 'Choose a new deployment output directory' >&2; exit 1; }
config="$repo/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
test -f "$config"
mkdir -p "$output"
for seed in 42 43; do
  if [[ "$seed" == 42 ]]; then source_run="$repo/results/B5-slow-public/CMP_v1";
  else source_run="$repo/results/B5-slow-public/CMP_seed43_review_v1"; fi
  model="$source_run/P/model.pt"
  python - "$model" "$seed" <<'PY'
import sys
from pathlib import Path
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read,sha
path=Path(sys.argv[1]); seed=int(sys.argv[2]); meta=read(path.with_suffix('.json')); bundle=load_bundle(path)
if sha(path)!=meta['bundle_sha256'] or meta['group']!='P' or meta.get('training_seed',42)!=seed:
    raise SystemExit('P export identity mismatch; keep existing files and inspect model.json')
if bundle.get('reference_score_protocol')!='mean-r-reliable-input-target-qc-v1':
    raise SystemExit('P package uses another score protocol')
print('P model verified; training seed',seed,'; reference',meta['score'],flush=True)
PY
  python -u -m baselines.B5_best.validate \
    --checkpoint "$model" --config "$config" \
    --device cuda --verify-reference \
    --output-dir "$output/seed$seed" \
    2>&1 | tee "$output/seed$seed.log"
done
python - "$output" <<'PY'
import sys
from pathlib import Path
from baselines.B5_best.common import read,write,csv_write
out=Path(sys.argv[1]); rows=[]
for seed in (42,43):
    report=read(out/f'seed{seed}/evaluation.json')
    if report['reference_matched'] is not True: raise SystemExit('Reference was not reproduced')
    rows.append({'training_seed':seed,**report['score'],'reference_matched':True})
csv_write(out/'summary.csv',rows); write(out/'summary.json',{'rows':rows,'complete':True})
print(rows)
PY
echo "Both P packages reproduced. Results: $output"
