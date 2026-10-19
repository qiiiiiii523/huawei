#!/usr/bin/env bash
# P frozen cache -> paired A/B O16 history preparation -> body-only training -> five groups.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
step=all
gpu=1
run=
config=
checkpoint=
baseline=
overlay=
old_body=
old_cache=
resume=0
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --step) step="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    --overlay) overlay="$2"; shift 2 ;;
    --old-body-checkpoint) old_body="$2"; shift 2 ;;
    --old-body-cache) old_cache="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
case "$step" in all|prepare|train-A|train-B|evaluate) ;; *) echo 'Use --step all|prepare|train-A|train-B|evaluate'; exit 2 ;; esac
[[ -n "$run" ]] || run="$repository/results/B5-best-context/P_body_AB_v1"
[[ -n "$checkpoint" ]] || checkpoint="$repository/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt"
[[ -n "$baseline" ]] || baseline="$repository/results/B5-best-context/P_O16_four_v1/base_inference"
[[ -n "$overlay" ]] || overlay="$(dirname "$repository")/task2_body_scale_ablation"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
[[ -n "$old_body" ]] || old_body="$repository/results/B5-best-context/body_verification_v1/o16/O16/best.pt"
[[ -n "$old_cache" ]] || old_cache="$repository/results/B5-best-context/body_clip16_v1/clip16_cache"
for file in "$checkpoint" "$baseline/evaluation.json" "$config" "$overlay/body_scale_ablation_manifest.json" "$old_body" "$old_cache/cache.json"; do
  test -f "$file" || { echo "Missing: $file" >&2; exit 1; }
done
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=setup
trap 'echo "Stopped: phase=$phase line=$LINENO; retain reports and repeat same step/settings with --resume: $run" >&2' ERR
# Only prepare writes the shared run identity. Parallel training reads it without racing.
if [[ "$step" == prepare || "$step" == all ]]; then
  if [[ -e "$run" && "$resume" == 0 ]]; then echo "Use new --run or --resume: $run" >&2; exit 1; fi
fi
python - "$run" "$checkpoint" "$baseline" "$config" "$overlay" "$old_body" "$old_cache" "$step" <<'PY'
from pathlib import Path
import sys
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.four_routes import SELECTED_BODY_SHA256
run, checkpoint, baseline, config, overlay, old, cache = map(Path, sys.argv[1:8])
step = sys.argv[8]
if sha(old) != SELECTED_BODY_SHA256:
    raise ValueError("Use selected old O16 seed42 best27")
report = read(baseline / "evaluation.json")
if report.get("completed") is not True or report["checkpoint_sha256"] != sha(checkpoint):
    raise ValueError("Baseline must be the completed P model.pt evaluation")
value = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint),
    baseline_evaluation=str(baseline.resolve()), baseline_report_sha256=sha(baseline / "evaluation.json"),
    config=str(config.resolve()), config_sha256=sha(config), overlay=str(overlay.resolve()),
    overlay_manifest_sha256=sha(overlay / "body_scale_ablation_manifest.json"),
    old_body_checkpoint=str(old.resolve()), old_body_sha256=sha(old),
    old_cache=str(cache.resolve()), old_cache_sha256=sha(cache / "cache.json"),
    seed=42, epochs=30, batch_size=16, learning_rate=.0002, body_alpha=1., encoding="legacy_beat16_stats8_v1")
if (run / "run.json").exists():
    if read(run / "run.json") != value:
        raise ValueError("A/B study inputs/settings changed; use a new run")
elif step in ("prepare", "all"):
    write(run / "run.json", value)
else:
    raise ValueError("First complete --step prepare")
PY
if [[ "$step" == prepare || "$step" == all ]]; then
  phase=P_training_cache
  extra=(); if [[ "$resume" == 1 ]]; then extra+=(--resume); fi
  python -u -m baselines.B5_best_context.body_prepare \
    --config "$config" --checkpoint "$checkpoint" --baseline-evaluation "$baseline" \
    --output "$run/P_cache" --device cuda "${extra[@]}" \
    2>&1 | tee -a "$run/prepare_P.log"
  phase=AB_history_features
  python -u -m baselines.B5_best_context.body_ab_prepare \
    --config "$config" --cache "$run/P_cache" --overlay "$overlay" \
    --output "$run/prepared" "${extra[@]}" \
    2>&1 | tee -a "$run/prepare_AB.log"
  for group in A B; do
    phase="preflight_$group"
    preflight="$run/preflight_$group"
    if [[ -f "$run/preflight_${group}_path.txt" ]]; then preflight=$(cat "$run/preflight_${group}_path.txt"); fi
    if [[ ! -f "$preflight/preflight.json" ]]; then
      if [[ -e "$preflight" ]]; then preflight="$run/preflight_${group}_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
      printf '%s\n' "$preflight" > "$run/preflight_${group}_path.txt"
      python -u -m baselines.B5_best_context.body_train \
        --cache "$run/prepared/${group}_cache" --output "$preflight" --group context \
        --seed 42 --epochs 30 --batch-size 16 --learning-rate .0002 --check-only \
        2>&1 | tee -a "$run/preflight_${group}.log"
    fi
    printf '%s\n' "$preflight" > "$run/preflight_${group}_path.txt"
  done
  python -u -m baselines.B5_best_context.body_ab_study \
    --config "$config" --prepared "$run/prepared" \
    --a-run "$(cat "$run/preflight_A_path.txt")" --b-run "$(cat "$run/preflight_B_path.txt")" \
    --old-body-checkpoint "$old_body" --old-body-cache "$old_cache" \
    --output "$run/preflight_pair.json" --check-only \
    2>&1 | tee -a "$run/preflight_pair.log"
fi
if [[ "$step" == train-A || "$step" == train-B || "$step" == all ]]; then
  test -f "$run/preflight_pair.json"
  groups=(A B)
  if [[ "$step" == train-A ]]; then groups=(A); fi
  if [[ "$step" == train-B ]]; then groups=(B); fi
  for group in "${groups[@]}"; do
    phase="train_$group"
    extra=()
    if [[ "$resume" == 1 && -f "$run/$group/config.json" ]]; then extra+=(--resume); fi
    python -u -m baselines.B5_best_context.body_train \
      --cache "$run/prepared/${group}_cache" --output "$run/$group" --group context \
      --seed 42 --epochs 30 --batch-size 16 --learning-rate .0002 --device cuda "${extra[@]}" \
      2>&1 | tee -a "$run/train_${group}.log"
  done
fi
if [[ "$step" == evaluate || "$step" == all ]]; then
  phase=evaluate
  study="$run/study"
  if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
  if [[ -f "$study/body_ab_study.json" ]]; then
    echo 'Completed study exists; use --resume to display it or a new --run for changed inputs.'
    [[ "$resume" == 1 ]] || exit 1
  else
    if [[ -e "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
    printf '%s\n' "$study" > "$run/study_path.txt"
    python -u -m baselines.B5_best_context.body_ab_study \
      --config "$config" --prepared "$run/prepared" --a-run "$run/A" --b-run "$run/B" \
      --old-body-checkpoint "$old_body" --old-body-cache "$old_cache" \
      --output "$study" --device cuda \
      2>&1 | tee -a "$run/evaluate.log"
  fi
  cat "$study/summary.csv"
fi
echo "Completed step=$step: $run"
