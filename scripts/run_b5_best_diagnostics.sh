#!/usr/bin/env bash
# Read saved evaluation arrays only. No git/network, GPU, checkpoint, or training.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repository_root=/home/qht/huawei
output=
extra=()
while (($#)); do
  case "$1" in
    --repository-root) repository_root=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
if [[ -z "$output" ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)_$$
  output="$repository_root/results/B5-best/diagnostics_${stamp}"
fi
if [[ -e "$output" ]]; then
  echo "Choose a new --output directory: $output" >&2
  exit 2
fi
mkdir -p -- "$output"
cd -- "$code_root"
# The Python entry requires an empty directory. Capture logs beside it instead.
echo "Code: $code_root"
echo "Existing data/results: $repository_root"
echo "Output: $output"
python -u -m baselines.B5_best.diagnose \
  --repository-root "$repository_root" \
  --output "$output" \
  "${extra[@]}" \
  2>&1 | tee "${output}.log"
if [[ -f "$output/report.md" ]]; then
  echo "Diagnostics finished: $output/report.md"
else
  echo "Preflight finished: $output/preflight.json"
fi
