#!/usr/bin/env bash
# Train ONLY O16; reuse the completed original-encoding O control.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
cache=
config=
control=
run=
gpu=1
resume=0
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --cache) cache="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --control-run) control="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$cache" ]] || cache="$repository/results/B5-best-context/body_v1/cache"
[[ -n "$control" ]] || control="$repository/results/B5-best-context/body_encoding_three_v1/O"
[[ -n "$run" ]] || run="$repository/results/B5-best-context/body_clip16_v1"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
test -f "$cache/cache.json"
test -f "$control/best.pt"
test -f "$control/config.json"
test -f "$control/completed.json"
test -f "$config"
if [[ "$resume" == 0 && -e "$run" ]]; then echo "Use a new --run or explicit --resume: $run" >&2; exit 1; fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=setup
trap 'echo "Stopped: phase=$phase line=$LINENO; repeat SAME settings with --resume: $run" >&2' ERR
python - "$run" "$cache" "$config" "$control" <<'PY'
from pathlib import Path
import sys
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.body_features import BODY_PROTOCOL, LEGACY_ENCODING, feature_encoding
run, cache, config, control = map(Path, sys.argv[1:])
cfg, done = read(control / "config.json"), read(control / "completed.json")
if (cfg.get("protocol") != BODY_PROTOCOL or cfg.get("group") != "context"
    or cfg.get("record_gate") is not None or feature_encoding(cfg) != LEGACY_ENCODING
    or cfg["cache_sha256"] != sha(cache / "cache.json")
    or done.get("completed") is not True or done["epochs"] != cfg["epochs"]
    or done["best_checkpoint_sha256"] != sha(control / "best.pt")):
    raise ValueError("Use a completed matched original-encoding O run, not C/G or an FM checkpoint")
value = dict(cache=str(cache.resolve()), cache_sha256=sha(cache / "cache.json"),
    config=str(config.resolve()), config_sha256=sha(config), control_run=str(control.resolve()),
    control_config_sha256=sha(control / "config.json"), control_checkpoint_sha256=sha(control / "best.pt"),
    seed=cfg["seed"], epochs=cfg["epochs"], batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"],
    initialization="same_random_weights_as_O", trained_group="O16", feature_encoding="legacy_beat16_stats8_v1")
if (run / "run.json").exists() and read(run / "run.json") != value:
    raise ValueError("Resume changed control/cache/config/budget; use a new run")
write(run / "run.json", value)
(run / "matched_settings.tsv").write_text("\t".join(str(cfg[k]) for k in ("seed", "epochs", "batch_size", "learning_rate")) + "\n", encoding="utf-8")
PY
IFS=$'\t' read -r seed epochs batch lr < "$run/matched_settings.tsv"
phase=prepare
extra=()
if [[ "$resume" == 1 ]]; then extra+=(--resume); fi
python -u -m baselines.B5_best_context.body_encoding_prepare \
  --config "$config" --cache "$cache" --encoding legacy_beat16_stats8_v1 \
  --output "$run/clip16_cache" "${extra[@]}" \
  2>&1 | tee -a "$run/prepare.log"
phase=preflight
preflight="$run/preflight_O16"
if [[ -f "$run/preflight_path.txt" ]]; then preflight=$(cat "$run/preflight_path.txt"); fi
if [[ ! -f "$preflight/preflight.json" ]]; then
  if [[ -d "$preflight" ]]; then preflight="$run/preflight_O16_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$run/clip16_cache" --group context --seed "$seed" --epochs "$epochs" \
    --batch-size "$batch" --learning-rate "$lr" --output "$preflight" --check-only \
    2>&1 | tee -a "$run/preflight.log"
fi
printf '%s\n' "$preflight" > "$run/preflight_path.txt"
phase=matched_initialization
python - "$control" "$preflight" "$cache" "$run" <<'PY'
from pathlib import Path
import sys
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.body_features import BEAT16_ENCODING
from baselines.B5_best_context.body_encoding_study import check_training_pair
control, preflight, cache, run = map(Path, sys.argv[1:])
check_training_pair(read(control / "config.json"), read(preflight / "config.json"),
    sha(cache / "cache.json"), sha(run / "clip16_cache/cache.json"), expected_encoding=BEAT16_ENCODING)
report = read(preflight / "preflight.json")
if not report.get("ready") or report.get("optimizer_updates") != 0:
    raise ValueError("Finish O16 CPU forward/backward checks first")
write(run / "preflight_pair.json", dict(ready=True, training_started=False, optimizer_updates=0,
    same_initial_weights=True, same_network_loss_budget_seed=True, groups=["O", "O16"]))
PY
phase=train_O16
extra=()
if [[ "$resume" == 1 && -f "$run/O16/config.json" ]]; then extra+=(--resume); fi
python -u -m baselines.B5_best_context.body_train \
  --cache "$run/clip16_cache" --group context --seed "$seed" --epochs "$epochs" \
  --batch-size "$batch" --learning-rate "$lr" --output "$run/O16" --device cuda "${extra[@]}" \
  2>&1 | tee -a "$run/O16.log"
phase=evaluation
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/encoding_checkpoints.json" ]]; then
  if [[ -d "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_encoding_study \
    --config "$config" --cache "$cache" --encoded-cache "$run/clip16_cache" \
    --control-run "$control" --encoded-run "$run/O16" --experimental-encoding legacy_beat16_stats8_v1 \
    --output "$study" --device cuda \
    2>&1 | tee -a "$run/study.log"
fi
printf '%s\n' "$study" > "$run/study_path.txt"
cat "$study/summary.csv"
cat "$study/encoding_effect.json"
echo "Completed: $run"
