#!/usr/bin/env bash
# Read-only fixed-alpha scan, independent of O/E/E0 encoding training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
checkpoint=
config=
run=
gpu=1
plots=()
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --no-plots) plots+=(--no-plots); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
cache="$repository/results/B5-best-context/body_v1/cache"
test -f "$config"
test -f "$cache/cache.json"
if [[ -z "$checkpoint" ]]; then
  checkpoint=$(python - "$repository" <<'PY'
import json, pathlib, sys
from baselines.B5_best.common import sha
root = pathlib.Path(sys.argv[1]) / "results/B5-best-context"
expected = "f23902346c786e0bef06388b86ef45c5514880f332ca1e0ff0e84c42f5688d45"
matches = []
for path in root.glob("body_C_extend30_*/C/best.pt"):
    completed = path.parent / "completed.json"
    if not completed.is_file():
        continue
    data = json.loads(completed.read_text(encoding="utf-8-sig"))
    if (data.get("completed") and data.get("record_gate") == "C" and
            data.get("best_epoch") == 26 and data.get("best_checkpoint_sha256") == expected and sha(path) == expected):
        matches.append(path.resolve())
if len(matches) != 1:
    raise SystemExit("Cannot uniquely find the reported C26 best.pt. Supply --checkpoint /absolute/path/C/best.pt (found %d)." % len(matches))
print(matches[0])
PY
  )
fi
test -f "$checkpoint"
[[ -n "$run" ]] || run="$repository/results/B5-best-context/body_alpha_C26_$(date +%Y%m%d_%H%M%S)_$$"
if [[ -e "$run" ]]; then echo "Use a new --run: $run" >&2; exit 1; fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Fixed C checkpoint: $checkpoint"
echo "Output: $run"
trap 'echo "Stopped: line=$LINENO; reports kept at $run; restart with a NEW --run" >&2' ERR
python -u -m baselines.B5_best_context.body_alpha_study \
  --config "$config" --cache "$cache" --checkpoint "$checkpoint" \
  --expected-epoch 26 --output "$run/study" --device cuda "${plots[@]}" \
  2>&1 | tee "$run/run.log"
cat "$run/study/summary.csv"
cat "$run/study/oracle_summary.json"
echo "Completed: $run"
