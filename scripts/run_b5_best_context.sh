#!/usr/bin/env bash
# One GPU base inference, then three fixed CPU postprocess groups. No git fetch or training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
gpu=1
root=
config=
checkpoint=
baseline=
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --output-dir) root="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use --gpu physical_GPU_number'; exit 2; }
[[ -z "$baseline" || -z "$checkpoint" ]] || { echo 'Use checkpoint or existing baseline, not both'; exit 2; }
if [[ -z "$root" ]]; then
  root="$repository/results/B5-best-context/fir25_$(date +%Y%m%d_%H%M%S)_$$"
fi
if [[ -z "$config" ]]; then
  config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
fi
test -f "$config"
[[ ! -e "$root" ]] || { echo "Use a new output directory: $root" >&2; exit 1; }
mkdir -p "$(dirname "$root")"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--config "$config" --output-dir "$root" --device cuda)
if [[ -n "$baseline" ]]; then args+=(--baseline-evaluation "$baseline"); fi
if [[ -n "$checkpoint" ]]; then args+=(--checkpoint "$checkpoint"); fi
# The log is adjacent to, not inside, the directory checked for emptiness.
python -u -m baselines.B5_best_context.validate "${args[@]}" 2>&1 | tee "${root}.log"
cat "$root/study/summary.csv"
echo "Finished; no training. Reports: $root/study"
