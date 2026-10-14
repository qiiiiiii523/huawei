#!/usr/bin/env bash
# Sequential single-GPU capability experiment; no git fetch/network/data downloads.
set -euo pipefail

repo=/home/qht/huawei
source_run=results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239
evaluation_dir=
gpu=1
epochs=50
batch_records=2
output=
resume=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo=$2; shift 2 ;;
    --source-run) source_run=$2; shift 2 ;;
    --evaluation-dir) evaluation_dir=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --epochs) epochs=$2; shift 2 ;;
    --batch-records) batch_records=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$repo"
if [[ -z "$evaluation_dir" ]]; then
  evaluation_dir="$source_run/evaluation_best_resume_20261012_173139_2232152"
fi
if [[ -z "$output" ]]; then
  output="results/B5-exp/slow_probe_$(date +%Y%m%d_%H%M%S)_$$"
fi
mkdir -p "$output"
output=$(cd "$output" && pwd)
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
phase=checks
trap 'status=$?; echo "Stopped: phase=$phase line=$LINENO exit=$status" >&2; echo "Output: $output" >&2; exit "$status"' ERR
printf 'GPU: %s\nOutput: %s\n' "$gpu" "$output"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
python -m unittest baselines.B5_exp.tests.test_probe baselines.B5_exp.tests.test_workflow -v 2>&1 | tee "$output/functional_tests.log"

phase=prepare
if [[ ! -f "$output/prepared/prepared.json" ]]; then
  python -m baselines.B5_exp.prepare \
    --checkpoint "$source_run/best.pt" \
    --evaluation-dir "$evaluation_dir" \
    --output "$output/prepared" \
    2>&1 | tee "$output/prepare.log"
fi
python -m baselines.B5_exp.check \
  --prepared "$output/prepared" \
  --report "$output/preflight.json" \
  2>&1 | tee "$output/preflight.log"
python -c 'import json,sys; from pathlib import Path; x=json.load(open(sys.argv[1])); assert Path(x["checkpoint"]).resolve()==Path(sys.argv[2]).resolve(), "Prepared data belong to another source run"' \
  "$output/prepared/prepared.json" "$source_run/best.pt"

for seconds in 10 30; do
  group="S${seconds}"
  run="$output/$group"
  mkdir -p "$run"
  phase="train_$group"
  if [[ -f "$run/last.pt" && "$resume" -ne 1 ]]; then
    echo "Existing checkpoints: rerun with --resume --output $output" >&2
    exit 2
  fi
  finished=0
  if [[ -f "$run/completed.json" ]]; then
    finished=$(python -c 'import json,sys; x=json.load(open(sys.argv[1])); print(int(x["completed"] and x["epochs"]==int(sys.argv[2])))' "$run/completed.json" "$epochs")
  fi
  if [[ "$finished" -ne 1 ]]; then
    extra=()
    if [[ -f "$run/last.pt" && "$resume" -eq 1 ]]; then
      extra=(--resume "$run/last.pt")
    fi
    python -u -m baselines.B5_exp.train \
      --prepared "$output/prepared" \
      --context-seconds "$seconds" \
      --output "$run" \
      --device cuda \
      --epochs "$epochs" \
      --batch-records "$batch_records" \
      --validate-every 5 \
      "${extra[@]}" \
      --execute-training \
      2>&1 | tee -a "$run/train.log"
  fi
  phase="evaluate_$group"
  if [[ ! -f "$run/evaluation_best/evaluation.json" ]]; then
    if [[ -d "$run/evaluation_best" ]]; then
      echo "Incomplete evaluation directory exists. Preserve/rename it before --resume: $run/evaluation_best" >&2
      exit 2
    fi
    python -u -m baselines.B5_exp.evaluate \
      --prepared "$output/prepared" \
      --checkpoint "$run/best.pt" \
      --device cuda \
      --output "$run/evaluation_best" \
      2>&1 | tee "$run/evaluation_best.log"
  else
    python -c 'import json,sys; from baselines.B5_exp.core import sha; x=json.load(open(sys.argv[1])); assert x["probe_checkpoint_sha256"]==sha(sys.argv[2]), "Evaluation is stale; rename it and evaluate again"' \
      "$run/evaluation_best/evaluation.json" "$run/best.pt"
  fi
done

phase=compare
if [[ ! -f "$output/comparison/audit.json" ]]; then
  python -m baselines.B5_exp.compare \
    --s10 "$output/S10/evaluation_best" \
    --s30 "$output/S30/evaluation_best" \
    --output "$output/comparison" \
    2>&1 | tee "$output/comparison.log"
else
  python -c 'import json,sys; from baselines.B5_exp.core import sha; x=json.load(open(sys.argv[1])); assert x["evaluation_json_sha256"]==[sha(p) for p in sys.argv[2:]], "Comparison is stale; rename the comparison directory and rerun"' \
    "$output/comparison/audit.json" "$output/S10/evaluation_best/evaluation.json" "$output/S30/evaluation_best/evaluation.json"
fi
printf 'Completed: %s/comparison/report.md\n' "$output"
