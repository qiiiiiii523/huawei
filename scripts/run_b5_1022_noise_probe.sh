#!/usr/bin/env bash
# Inference-only independent/antithetic K16 evaluation for the complete B5-1022 route.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repository=/home/qht/huawei
gpu=1
mode=
scale=1.0
noise_seed=42
slow_run=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/shared
g120=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/G120/best.pt
c3_cache=/home/qht/huawei/results/B5-1022/g120_c3_seed42_20261022_195813/cache
c3="$root/models/body_c3_seed42_best.pt"
output=

while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --noise-scale) scale="$2"; shift 2 ;;
    --noise-seed) noise_seed="$2"; shift 2 ;;
    --slow-run) slow_run="$2"; shift 2 ;;
    --g120-checkpoint) g120="$2"; shift 2 ;;
  --c3-cache) c3_cache="$2"; shift 2 ;;
    --c3-checkpoint) c3="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "--gpu must be a physical GPU index" >&2; exit 2; }
[[ "$mode" == independent || "$mode" == antithetic ]] || { echo "--mode independent|antithetic is required" >&2; exit 2; }
for value in "$slow_run" "$g120" "$c3_cache" "$c3"; do
  [[ -n "$value" ]] || { echo "A required B5-1022 path is empty" >&2; exit 2; }
done
for value in "$slow_run/study.json" "$g120" "$c3_cache/cache.json" "$c3"; do
  [[ -f "$value" ]] || { echo "Required B5-1022 asset is missing: $value" >&2; exit 1; }
done
[[ -n "$output" ]] || output="$repository/results/B5-1022/noise_probes/${mode}_scale${scale//./p}_seed${noise_seed}_$(date +%Y%m%d_%H%M%S)"
[[ ! -e "$output" ]] || { echo "Use a new output directory: $output" >&2; exit 1; }

cd "$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
python -u scripts/b5_1022_noise_probe.py \
  --repository-root "$repository" \
  --slow-run "$slow_run" \
  --g120-checkpoint "$g120" \
  --c3-cache "$c3_cache" \
  --c3-checkpoint "$c3" \
  --output "$output" \
  --mode "$mode" \
  --noise-scale "$scale" \
  --noise-seed "$noise_seed" \
  --device cuda \
  2>&1 | tee "$output.log"

cat "$output/base_evaluation/correlation_score.csv"
cat "$output/final/summary.csv"
printf 'Completed: %s\n' "$output"
