#!/usr/bin/env bash
# Three serial stages; real batch64, no initial/intermediate validation, GPU2.
set -euo pipefail
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CODE"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
PYTHON_EXE=${PYTHON_EXE:-/home/qht/.conda/envs/BioFlow/bin/python}
RUN=${RUN:-"$CODE/results/FM_Unet_M7/seed42_$(date +%Y%m%d_%H%M%S)"}
PUBLIC_MANIFEST=${PUBLIC_MANIFEST:-/home/qht/huawei/data_cache/B7_public_qcv2/data/public10/manifest.json}
PTB_MANIFEST=${PTB_MANIFEST:-/home/qht/huawei_B7/results/B7/auto_seed42/ptb_recovery/data/manifest.json}
HUAWEI_MANIFEST=${HUAWEI_MANIFEST:-/home/qht/huawei_B7/results/B7/auto_seed42/data/huawei/manifest.json}
FAST_CODE=${FAST_CODE:-/home/qht/huawei_B5_final}
FAST_CHECKPOINT=${FAST_CHECKPOINT:-/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt}
DEMOGRAPHICS=${DEMOGRAPHICS:-/home/qht/huawei/data/userinfobean.csv}
mkdir -p "$RUN/logs" "$CODE/results/FM_Unet_M7"
printf '%s\n' "$RUN" > "$CODE/results/FM_Unet_M7/latest_run.txt"
printf '[pipeline] RUN=%s GPU=2 real_batch=64 accumulation=1 budgets=2000/4000/6000 final_validation_only\n' "$RUN"
for path in "$PUBLIC_MANIFEST" "$PTB_MANIFEST" "$HUAWEI_MANIFEST" "$FAST_CHECKPOINT" "$DEMOGRAPHICS"; do
    test -f "$path" || { printf 'Missing required file: %s\n' "$path" >&2; exit 1; }
done

run_stage() {
    local stage=$1 manifest=$2 previous=${3:-}
    if test -f "$RUN/$stage/completed.json"; then
        printf '[pipeline] already completed %s\n' "$stage"
        return
    fi
    local args=( -u -m FM_Unet.train --manifest "$manifest"
        --config "configs/fm_unet_m7/$stage.json" --output "$RUN/$stage"
        --cache-dir "$RUN/cache" --device cuda:0 --cpu-threads 4 )
    if test -f "$RUN/$stage/last.pt"; then
        args+=( --resume )
    elif test -n "$previous"; then
        args+=( --init-checkpoint "$RUN/$previous/final.pt" )
    fi
    if test "$stage" = huawei120; then
        args+=( --fast-code "$FAST_CODE" --fast-checkpoint "$FAST_CHECKPOINT"
            --demographics-csv "$DEMOGRAPHICS" )
    fi
    "$PYTHON_EXE" "${args[@]}" 2>&1 | tee -a "$RUN/logs/$stage.log"
    test -f "$RUN/$stage/completed.json"
}

run_stage public10 "$PUBLIC_MANIFEST"
run_stage public75 "$PTB_MANIFEST" public10
run_stage huawei120 "$HUAWEI_MANIFEST" public75
printf '[pipeline] COMPLETE %s\n' "$RUN"
