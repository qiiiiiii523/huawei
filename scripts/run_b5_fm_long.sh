#!/usr/bin/env bash
# Default: prepare and screen only. Actual optimizer updates require --train.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
gpu=1
run=
checkpoint=
config=
source_fm=
experiment=configs/b5_fm_long.json
batch_size=1
train=0
existing_shared=
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --experiment-config) experiment="$2"; shift 2 ;;
    --fm-source-checkpoint) source_fm="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --shared) existing_shared="$2"; shift 2 ;;
    --train) train=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Physical integer GPU required' >&2; exit 2; }
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo 'Positive batch size required' >&2; exit 2; }
[[ -n "$run" ]] || run="$repository/results/B5-final-exp-FM/run_$(date +%Y%m%d_%H%M%S)_$$"
[[ "$run" == /* ]] || { echo 'Use an absolute --run path' >&2; exit 2; }
[[ ! -e "$run" ]] || { echo "Use a NEW result directory: $run" >&2; exit 1; }
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
phase=prepare
trap 'echo "Stopped: phase=$phase line=$LINENO; reports retained at $run. See docs/b5_fm_long.md." >&2' ERR
if [[ -n "$existing_shared" ]]; then
  [[ "$existing_shared" == /* ]] || { echo 'Use an absolute --shared path' >&2; exit 2; }
  # Link only the explicitly supplied immutable study; do not copy datasets.
  ln -s "$existing_shared" "$run/shared"
else
  args=(--repository-root "$repository" --experiment-config "$experiment" --output "$run/shared" --device cuda)
  [[ -z "$checkpoint" ]] || args+=(--checkpoint "$checkpoint")
  [[ -z "$config" ]] || args+=(--config "$config")
  [[ -z "$source_fm" ]] || args+=(--fm-source-checkpoint "$source_fm")
  python -u -m baselines.B5_fm_long.prepare "${args[@]}" 2>&1 | tee "$run/prepare.log"
fi
for seconds in 10 40; do
  phase="screen_FM$seconds"
  python -u -m baselines.B5_fm_long.evaluate --run "$run/shared" --window-seconds "$seconds" \
    --output "$run/screen/FM$seconds" --device cuda --batch-size "$batch_size" \
    2>&1 | tee "$run/screen_FM$seconds.log"
done
if [[ "$train" == 0 ]]; then
  phase=summarize_screen
  python -m baselines.B5_fm_long.summarize --run "$run" --output "$run/comparison"
  echo "Screening completed; ZERO optimizer updates: $run/comparison/summary.md"
  exit 0
fi
for seconds in 10 40; do
  phase="train_FM$seconds"
  python -u -m baselines.B5_fm_long.train --run "$run/shared" --window-seconds "$seconds" \
    --output "$run/FM$seconds" --device cuda --batch-size "$batch_size" --execute-training \
    2>&1 | tee "$run/train_FM$seconds.log"
  for candidate in best_raw best_hybrid last; do
    phase="verify_FM${seconds}_$candidate"
    python -u -m baselines.B5_fm_long.evaluate --run "$run/shared" --candidate "$run/FM$seconds/$candidate.pt" \
      --output "$run/verification/FM$seconds/$candidate" --device cuda --batch-size "$batch_size" \
      2>&1 | tee "$run/verify_FM${seconds}_$candidate.log"
  done
  for mode in raw hybrid; do
    phase="package_FM${seconds}_$mode"
    python -u -m baselines.B5_fm_long.predict package --run "$run/shared" \
      --candidate "$run/FM$seconds/best_$mode.pt" --output "$run/packages/FM${seconds}_$mode.pt" \
      2>&1 | tee "$run/package_FM${seconds}_$mode.log"
    phase="verify_package_FM${seconds}_$mode"
    python -u -m baselines.B5_fm_long.evaluate --run "$run/shared" --package "$run/packages/FM${seconds}_$mode.pt" \
      --output "$run/verification/FM$seconds/package_$mode" --device cuda --batch-size "$batch_size" \
      2>&1 | tee "$run/verify_package_FM${seconds}_$mode.log"
  done
done
phase=summarize
python -m baselines.B5_fm_long.summarize --run "$run" --output "$run/comparison"
echo "Completed: $run/comparison/summary.md"
