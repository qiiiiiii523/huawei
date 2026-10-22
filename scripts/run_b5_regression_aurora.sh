#!/usr/bin/env bash
set -euo pipefail

ACTION="$1"
HUAWEI_CONFIG="configs/experiments/b5_main_v2_public_finetune_meta.yaml"
EXPERIMENT_CONFIG="configs/b5_regression_aurora.json"

case "$ACTION" in
prepare)
  python -m baselines.B5_regression_aurora.prepare --ptbxl-root "$PUBLIC_PTBXL" --chapman-root "$PUBLIC_CHAPMAN" --huawei-config "$HUAWEI_CONFIG" --scales "$SCALES" --output "$STUDY"
  ;;
public)
  python -m baselines.B5_regression_aurora.train --stage public --experiment-config "$EXPERIMENT_CONFIG" --study "$STUDY" --output "$OUTPUT" --device "$DEVICE"
  ;;
huawei)
  python -m baselines.B5_regression_aurora.train --stage huawei --experiment-config "$EXPERIMENT_CONFIG" --huawei-config "$HUAWEI_CONFIG" --init-checkpoint "$PUBLIC_CHECKPOINT" --output "$OUTPUT" --device "$DEVICE"
  ;;
evaluate)
  python -m baselines.B5_regression_aurora.evaluate --experiment-config "$EXPERIMENT_CONFIG" --huawei-config "$HUAWEI_CONFIG" --checkpoint "$HUAWEI_CHECKPOINT" --output "$OUTPUT" --device "$DEVICE"
  ;;
*)
  echo "first argument must be prepare, public, huawei, or evaluate" >&2
  exit 2
  ;;
esac
