#!/usr/bin/env bash
# Evaluate the frozen selected composition; no training, fetching, or source changes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
gpu=1
run=
config=
baseline=
regenerate=0
asset_args=()
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    --profile|--checkpoint|--body-checkpoint) asset_args+=("$1" "$2"); shift 2 ;;
    --regenerate-base) regenerate=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical integer index' >&2; exit 2; }
if [[ "$regenerate" == 1 && -n "$baseline" ]]; then
  echo 'Choose either --regenerate-base or --baseline-evaluation' >&2; exit 2
fi
[[ -n "$run" ]] || run="$repository/results/B5-final/evaluation_$(date +%Y%m%d_%H%M%S)_$$"
if [[ -e "$run" ]]; then echo "Use a new --run directory: $run" >&2; exit 1; fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=verify_selected_weights
trap 'echo "Stopped: phase=$phase line=$LINENO. Reports retained: $run. Use a new --run for retry." >&2' ERR
python -u -m baselines.B5_final.profile \
  --repository-root "$repository" "${asset_args[@]}" --report "$run/preflight.json" \
  2>&1 | tee "$run/preflight.log"
if [[ -z "$config" ]]; then
  config=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["validation_config"])' "$run/preflight.json")
fi
test -f "$config"
if [[ "$regenerate" == 1 ]]; then
  phase=generate_plain_FM_P_predictions
  checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_checkpoint"])' "$run/preflight.json")
  baseline="$run/base_inference"
  python -u -m baselines.B5_best.validate \
    --checkpoint "$checkpoint" --config "$config" --output-dir "$baseline" --device cuda \
    2>&1 | tee "$run/base_inference.log"
elif [[ -z "$baseline" ]]; then
  baseline=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_evaluation"])' "$run/preflight.json")
fi
test -f "$baseline/evaluation.json"
phase=evaluate_four_context_routes
python -u -m baselines.B5_final.validate \
  --repository-root "$repository" "${asset_args[@]}" \
  --config "$config" --baseline-evaluation "$baseline" \
  --output "$run/study" --device cuda --verify-reference \
  2>&1 | tee "$run/evaluate.log"
cat "$run/study/summary.csv"
echo "Completed without training. Final result: $run/study/final.json"
echo "Results directory: $run"
