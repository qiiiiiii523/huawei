#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ $# -ge 2 ]] || { echo 'Usage: bash scripts/run_b5_context_public.sh prepare|references|control|context_public|compare RUN [GPU] [prepare args...]'; exit 2; }
phase=$1
run=$2
gpu=${3:-0}
shift 2
if (($#)); then shift; fi
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Physical GPU index must be integer'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
mkdir -p "$run"
trap 'echo "Stopped phase=$phase line=$LINENO exit=$?; keep results at $run" >&2' ERR
archive_partial_evaluation() {
  # Preserve interrupted report directories so a retry gets a clean destination.
  local folder=$1
  if [[ -d "$folder" && ! -f "$folder/evaluation.json" ]]; then
    mv -- "$folder" "${folder}.interrupted_$(date +%Y%m%d_%H%M%S)_$$"
  fi
}
case "$phase" in
  prepare)
    python -u -m baselines.B5_context_public.prepare --output "$run/shared" --device cuda "$@" 2>&1 | tee "$run/prepare.log"
    ;;
  references)
    python -u -m baselines.B5_context_public.references --run "$run/shared" --output "$run/references" --device cuda 2>&1 | tee "$run/references.log"
    ;;
  control|context_public)
    resume=()
    if [[ -f "$run/$phase/last.pt" && ! -f "$run/$phase/completed.json" ]]; then
      resume=(--resume "$run/$phase/last.pt")
    fi
    if [[ ! -f "$run/$phase/completed.json" ]]; then
      python -u -m baselines.B5_context_public.train --run "$run/shared" --references "$run/references/references.json" \
        --arm "$phase" --output "$run/$phase" --device cuda --execute-training "${resume[@]}" 2>&1 | tee -a "$run/${phase}.log"
    fi
    for candidate in best_r last; do
      if [[ ! -f "$run/$phase/evaluations/$candidate/evaluation.json" ]]; then
        archive_partial_evaluation "$run/$phase/evaluations/$candidate"
        python -u -m baselines.B5_context_public.evaluate --run "$run/shared" --checkpoint "$run/$phase/$candidate.pt" \
          --output "$run/$phase/evaluations/$candidate" --device cuda 2>&1 | tee -a "$run/${phase}_eval.log"
      fi
    done
    if [[ "$phase" == context_public ]]; then
      for variant in shuffled disabled; do
        if [[ ! -f "$run/$phase/diagnostics/$variant/evaluation.json" ]]; then
          archive_partial_evaluation "$run/$phase/diagnostics/$variant"
          python -u -m baselines.B5_context_public.evaluate --run "$run/shared" --checkpoint "$run/$phase/best_r.pt" \
            --output "$run/$phase/diagnostics/$variant" --history "$variant" --device cuda 2>&1 | tee -a "$run/${phase}_diagnostics.log"
        fi
      done
    fi
    ;;
  compare)
    python -m baselines.B5_context_public.compare --run "$run"
    ;;
  *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
