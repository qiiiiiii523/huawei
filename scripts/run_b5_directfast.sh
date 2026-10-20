#!/usr/bin/env bash
# Run on the server. Creates only a NEW experiment result directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
gpu=1
run=
config=
baseline=
experiment=configs/b5_directfast.json
source_fm=
batch_size=4
prepare_only=0
regenerate=0
asset_args=()
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    --experiment-config) experiment="$2"; shift 2 ;;
    --fm-source-checkpoint) source_fm="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --profile|--checkpoint|--body-checkpoint) asset_args+=("$1" "$2"); shift 2 ;;
    --prepare-only) prepare_only=1; shift ;;
    --regenerate-base) regenerate=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical integer index' >&2; exit 2; }
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo 'Positive batch size required' >&2; exit 2; }
[[ "$regenerate" == 0 || -z "$baseline" ]] || { echo 'Choose regenerate OR existing baseline' >&2; exit 2; }
[[ -n "$run" ]] || run="$repository/results/B5-final-exp-directfast/run_$(date +%Y%m%d_%H%M%S)_$$"
[[ "$run" == /* ]] || { echo 'Use an absolute --run path' >&2; exit 2; }
[[ ! -e "$run" ]] || { echo "Use a NEW run directory: $run" >&2; exit 1; }
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
phase=assets
trap 'echo "Stopped: phase=$phase line=$LINENO. Reports retained: $run. See docs/b5_directfast.md." >&2' ERR
python -u -m baselines.B5_final.profile --repository-root "$repository" "${asset_args[@]}" \
  --report "$run/assets.json" 2>&1 | tee "$run/assets.log"
[[ -n "$config" ]] || config=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["validation_config"])' "$run/assets.json")
if [[ "$regenerate" == 1 ]]; then
  phase=regenerate_original_base
  checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_checkpoint"])' "$run/assets.json")
  baseline="$run/base_inference"
  python -u -m baselines.B5_best.validate --checkpoint "$checkpoint" --config "$config" \
    --output-dir "$baseline" --device cuda --batch-size "$batch_size" 2>&1 | tee "$run/base_inference.log"
fi
[[ -n "$baseline" ]] || baseline=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_evaluation"])' "$run/assets.json")
phase=reproduce_Final
python -u -m baselines.B5_final.validate --repository-root "$repository" "${asset_args[@]}" \
  --config "$config" --baseline-evaluation "$baseline" --output "$run/G0" --device cuda \
  --verify-reference 2>&1 | tee "$run/G0.log"
phase=prepare
prepare_args=()
[[ -z "$source_fm" ]] || prepare_args+=(--fm-source-checkpoint "$source_fm")
python -u -m baselines.B5_directfast.prepare --repository-root "$repository" "${asset_args[@]}" \
  --config "$config" --experiment-config "$experiment" --baseline-evaluation "$baseline" \
  --g0-report "$run/G0/final.json" --output "$run/shared" --device cuda "${prepare_args[@]}" \
  2>&1 | tee "$run/prepare.log"
phase=server_preflight
python -u -m baselines.B5_directfast.check --run "$run/shared" --output "$run/server_preflight.json" \
  --device cuda 2>&1 | tee "$run/server_preflight.log"
phase=evaluate_F0
python -u -m baselines.B5_directfast.evaluate --run "$run/shared" --output "$run/F0" \
  --device cuda --batch-size "$batch_size" 2>&1 | tee "$run/F0.log"
if [[ "$prepare_only" == 1 ]]; then
  echo "Prepared and checked on SERVER, zero optimizer updates: $run"
  exit 0
fi
for group in F1 D1; do
  phase="train_$group"
  python -u -m baselines.B5_directfast.train --run "$run/shared" --group "$group" \
    --output "$run/$group" --device cuda --batch-size "$batch_size" --execute-training \
    2>&1 | tee "$run/$group.log"
done
for group in F1 D1; do
  for candidate in best last; do
    phase="verify_${group}_$candidate"
    python -u -m baselines.B5_directfast.evaluate --run "$run/shared" --candidate "$run/$group/$candidate.pt" \
      --output "$run/verification/$group/$candidate" --device cuda --batch-size "$batch_size" \
      2>&1 | tee "$run/verify_${group}_$candidate.log"
  done
  phase="package_$group"
  python -u -m baselines.B5_directfast.package --run "$run/shared" --candidate "$run/$group/best.pt" \
    --output "$run/packages/${group}_best.pt" 2>&1 | tee "$run/package_$group.log"
  phase="verify_package_$group"
  python -u -m baselines.B5_directfast.evaluate --run "$run/shared" --package "$run/packages/${group}_best.pt" \
    --output "$run/verification/$group/package" --device cuda --batch-size "$batch_size" \
    2>&1 | tee "$run/verify_package_$group.log"
done
phase=summarize
python -u -m baselines.B5_directfast.summarize --run "$run" --output "$run/comparison"
echo "Completed: $run/comparison/summary.md"
