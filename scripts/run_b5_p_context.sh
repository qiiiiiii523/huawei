#!/usr/bin/env bash
# Fixed P bundle -> one full baseline generation -> four frozen context routes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
checkpoint=
body=
cache=
config=
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
    --run) run="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$checkpoint" ]] || checkpoint="$repository/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt"
[[ -n "$body" ]] || body="$repository/results/B5-best-context/body_verification_v1/o16/O16/best.pt"
[[ -n "$cache" ]] || cache="$repository/results/B5-best-context/body_clip16_v1/clip16_cache"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
[[ -n "$run" ]] || run="$repository/results/B5-best-context/P_O16_four_v1"
for file in "$checkpoint" "$body" "$cache/cache.json" "$config"; do test -f "$file" || { echo "Missing: $file" >&2; exit 1; }; done
if [[ -e "$run" && "$resume" == 0 ]]; then echo "Use a new --run or explicit --resume: $run" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=setup
trap 'echo "Stopped: phase=$phase line=$LINENO; retain reports, repeat SAME --run with --resume: $run" >&2' ERR
python - "$run" "$checkpoint" "$body" "$cache" "$config" <<'PY'
from pathlib import Path
import sys
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.four_routes import SELECTED_BODY_SHA256
run, base, body, cache, config = map(Path, sys.argv[1:])
if sha(body) != SELECTED_BODY_SHA256:
    raise ValueError("Use the selected O16 seed42 best27 body weight")
value = dict(checkpoint=str(base.resolve()), checkpoint_sha256=sha(base),
    body_checkpoint=str(body.resolve()), body_checkpoint_sha256=sha(body),
    body_cache=str(cache.resolve()), body_cache_sha256=sha(cache / "cache.json"),
    config=str(config.resolve()), config_sha256=sha(config),
    sampling=dict(seed=42, solver="heun", steps=16, samples=16),
    slow_beta=.75, body_alpha=1., allow_base_transfer=True, training_started=False)
if (run / "run.json").exists() and read(run / "run.json") != value:
    raise ValueError("Resume changed model/context/data/settings; use a new run")
write(run / "run.json", value)
PY
phase=base_generation
if [[ ! -f "$run/base_inference/evaluation.json" ]]; then
  if [[ -e "$run/base_inference" ]]; then
    echo "Incomplete baseline retained. Choose a new --run; no partial arrays are reused." >&2
    exit 1
  fi
  python -u -m baselines.B5_best.validate \
    --checkpoint "$checkpoint" --config "$config" \
    --output-dir "$run/base_inference" --device cuda \
    2>&1 | tee "$run/base_inference.log"
fi
phase=four_routes
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/four_routes.json" ]]; then
  if [[ -e "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  printf '%s\n' "$study" > "$run/study_path.txt"
  python -u -m baselines.B5_best_context.four_routes \
    --config "$config" --checkpoint "$checkpoint" \
    --baseline-evaluation "$run/base_inference" \
    --body-checkpoint "$body" --body-cache "$cache" \
    --allow-base-transfer --output "$study" --device cuda \
    2>&1 | tee -a "$run/four_routes.log"
fi
cat "$study/summary.csv"
echo "Finished without training: $run"
echo "Four-group results: $study/summary.csv"
