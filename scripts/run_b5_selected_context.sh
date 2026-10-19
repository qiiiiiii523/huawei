#!/usr/bin/env bash
# One complete base generation + baseline/d6/selected-B43 evaluation; no training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
checkpoint=
body=
cache=
config=
baseline=
run=
gpu=1
resume=0
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --body-checkpoint) body="$2"; shift 2 ;;
    --body-cache) cache="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$checkpoint" ]] || checkpoint="$repository/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt"
[[ -n "$body" ]] || body="$repository/results/B5-best-context/P_body_AB_seed43_v1/B/best.pt"
[[ -n "$cache" ]] || cache="$repository/results/B5-best-context/P_body_AB_v1/prepared/B_cache"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
[[ -n "$run" ]] || run="$repository/results/B5-best-context/FM_P_best45_B43_v1"
for file in "$checkpoint" "$body" "$cache/cache.json" "$config"; do test -f "$file" || { echo "Missing: $file" >&2; exit 1; }; done
if [[ -e "$run" && "$resume" == 0 ]]; then echo "Use new --run or --resume: $run" >&2; exit 1; fi
if [[ -n "$baseline" ]]; then test -f "$baseline/evaluation.json"; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=setup
trap 'echo "Stopped: phase=$phase line=$LINENO; retain reports and repeat SAME settings with --resume: $run" >&2' ERR
python - "$run" "$checkpoint" "$body" "$cache" "$config" "$baseline" <<'PY'
from pathlib import Path
import sys
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.selected_study import SELECTED_B43_SHA256
from baselines.B5_best_context.body_history import B_POLICY, history_policy
run, base, body, cache, config = map(Path, sys.argv[1:6])
baseline = None if not sys.argv[6] else Path(sys.argv[6]).resolve()
if sha(body) != SELECTED_B43_SHA256 or history_policy(read(cache/"cache.json")) != B_POLICY:
    raise ValueError("Use selected B-seed43-best15 with its original B cache")
value = dict(checkpoint=str(base.resolve()), checkpoint_sha256=sha(base),
    body_checkpoint=str(body.resolve()), body_checkpoint_sha256=sha(body),
    body_cache=str(cache.resolve()), body_cache_sha256=sha(cache/"cache.json"),
    config=str(config.resolve()), config_sha256=sha(config),
    baseline_evaluation=None if baseline is None else str(baseline),
    baseline_report_sha256=None if baseline is None else sha(baseline/"evaluation.json"),
    sampling=dict(seed=42, solver="heun", steps=16, samples=16), slow_beta=.75,
    body_alpha=1., history_preprocessing=B_POLICY, allow_base_transfer=True, training_started=False)
if (run/"run.json").exists() and read(run/"run.json") != value:
    raise ValueError("Resume changed model/context/data/settings; use a new run")
write(run/"run.json", value)
PY
phase=base_generation
if [[ -z "$baseline" ]]; then
  baseline="$run/base_inference"
  if [[ ! -f "$baseline/evaluation.json" ]]; then
    if [[ -e "$baseline" ]]; then echo 'Incomplete base generation retained; use a new --run.' >&2; exit 1; fi
    python -u -m baselines.B5_best.validate \
      --checkpoint "$checkpoint" --config "$config" --output-dir "$baseline" --device cuda \
      2>&1 | tee "$run/base_inference.log"
  fi
fi
phase=selected_context
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/selected_context.json" ]]; then
  if [[ -e "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  printf '%s\n' "$study" > "$run/study_path.txt"
  python -u -m baselines.B5_best_context.selected_study \
    --config "$config" --checkpoint "$checkpoint" --baseline-evaluation "$baseline" \
    --body-checkpoint "$body" --body-cache "$cache" --allow-base-transfer \
    --output "$study" --device cuda \
    2>&1 | tee -a "$run/selected_context.log"
fi
cat "$study/summary.csv"
echo "Completed without training: $run"
