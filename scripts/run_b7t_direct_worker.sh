#!/usr/bin/env bash
set -euo pipefail
preset="$1"
case "$preset" in g6_19m|w144_g5_21m) ;; *) exit 2 ;; esac
cd "$B7T_ROOT"
exec "$B7T_PYTHON" -u -m baselines.B7.direct_suite \
  --preset "$preset" \
  --source-run "$B7T_SOURCE/$preset/seed42" \
  --huawei-manifest "$B7T_HUAWEI" --ptb-manifest "$B7T_PTB" \
  --output "$B7T_OUTPUT/$preset/seed42" \
  --gpu "$B7T_GPU" --batch-size 16 --eval-batch-size 4 \
  --updates 1000 --q-updates 1000 \
  >> "$B7T_OUTPUT/$preset/seed42/launcher.log" 2>&1
