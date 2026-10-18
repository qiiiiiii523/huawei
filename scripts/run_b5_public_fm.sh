#!/usr/bin/env bash
# No git/network operations. All data/results stay in the user's original repository.
set -euo pipefail

action=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
repo=/home/qht/huawei
run=
gpu=0
group=
stage=both
resume=0
ptb=
chapman=
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository-root) repo=$2; shift 2 ;;
    --run) run=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --group) group=$2; shift 2 ;;
    --stage) stage=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    --resume-audit) extra+=(--resume-audit); shift ;;
    --ptbxl-root) ptb=$2; shift 2 ;;
    --chapman-root) chapman=$2; shift 2 ;;
    --public-checkpoint|--fixed-bundle|--data-reference|--data-config|--settings|--chapman-probability|--chapman-patient-map)
      extra+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code"
run=${run:-$repo/results/B5-best-exp-PublicFM/public_expansion_v1}
ptb=${ptb:-$repo/data/PTB-XL}
chapman=${chapman:-$repo/data/Chapman_Shaoxing_Ningbo}
if [[ ! $gpu =~ ^[0-9]+$ ]]; then echo "--gpu must be one physical GPU index" >&2; exit 2; fi
export PYTHONUNBUFFERED=1

case "$action" in
  prepare)
    mkdir -p "$run"
    python -u -m baselines.B5_public_fm.prepare \
      --repository-root "$repo" --ptbxl-root "$ptb" --chapman-root "$chapman" \
      --output "$run/shared" "${extra[@]}" 2>&1 | tee "$run/prepare.log"
    ;;
  check)
    python -u -m baselines.B5_public_fm.check \
      --study "$run/shared/study.json" --report "$run/preflight.json" \
      2>&1 | tee "$run/preflight.log"
    ;;
  baseline)
    CUDA_VISIBLE_DEVICES=$gpu python -u -m baselines.B5_public_fm.evaluate \
      --study "$run/shared/study.json" --device cuda --output "$run/baseline" \
      2>&1 | tee "$run/baseline.log"
    ;;
  train)
    if [[ $group != control && $group != expanded ]]; then echo "Use --group control or expanded" >&2; exit 2; fi
    if [[ $stage != public && $stage != huawei && $stage != both ]]; then echo "Use --stage public, huawei or both" >&2; exit 2; fi
    flags=()
    if [[ $resume == 1 ]]; then flags+=(--resume); fi
    CUDA_VISIBLE_DEVICES=$gpu python -u -m baselines.B5_public_fm.train \
      --study "$run/shared/study.json" --group "$group" --stage "$stage" \
      --output "$run/$group" --device cuda --execute-training "${flags[@]}" \
      2>&1 | tee -a "$run/${group}.log"
    ;;
  compare)
    python -u -m baselines.B5_public_fm.compare \
      --study "$run/shared/study.json" --baseline "$run/baseline" \
      --control "$run/control" --expanded "$run/expanded" --output "$run/comparison" \
      2>&1 | tee "$run/compare.log"
    ;;
  pack)
    python -m baselines.B5_public_fm.pack --run "$run" --output "$repo/B5_PublicFM_reports.tar.gz"
    ;;
  help|-h|--help)
    echo 'Usage: bash scripts/run_b5_public_fm.sh prepare|check|baseline|train|compare|pack [options]'
    echo 'prepare: --ptbxl-root PATH --chapman-root PATH [--public-checkpoint PATH] [--fixed-bundle PATH] [--resume-audit]'
    echo 'train: --group control|expanded --gpu N [--stage public|huawei|both] [--resume]'
    echo 'Common: --run ABSOLUTE_RUN_ROOT --repository-root /home/qht/huawei'
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
