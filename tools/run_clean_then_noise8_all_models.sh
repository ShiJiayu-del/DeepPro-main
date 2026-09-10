#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

CLEAN_DATA="$REPO_ROOT/../datasets_v1/NUDT-MIRSDT"
CLEAN_CONTROL="$REPO_ROOT/experiments/nudt_mirsdt_all_models_2026-09-01"
CLEAN_SAVE="$REPO_ROOT/log/sem_seg/_queues/nudt_mirsdt_all_models_2026-09-01"
NOISE_DATA="$REPO_ROOT/../datasets_v1/NUDT-MIRSDT-Noise8.0_FJY"
NOISE_CONTROL="$REPO_ROOT/experiments/nudt_mirsdt_noise8_fjy_all_models_2026-09-03"
NOISE_SAVE="$REPO_ROOT/log/sem_seg/_queues/nudt_mirsdt_noise8_fjy_all_models_2026-09-03"
PIPELINE_ROOT="$REPO_ROOT/log/sem_seg/_pipeline"
PIPELINE_LOG="$PIPELINE_ROOT/clean_then_noise8_all_models_2026-09-03.log"

mkdir -p "$PIPELINE_ROOT"
exec 8>"$PIPELINE_ROOT/.clean_then_noise8_all_models.lock"
if ! flock -n 8; then
    echo "The clean-to-Noise8 pipeline is already running." >&2
    exit 1
fi
exec > >(tee -a "$PIPELINE_LOG") 2>&1

all_done() {
    local manifest="$1"
    local save_root="$2"
    local expected=0
    local run_id model variant loss log_dir
    while IFS=$'\t' read -r run_id model variant loss log_dir; do
        [[ "$run_id" == "run_id" || -z "$run_id" ]] && continue
        expected=$((expected + 1))
        [[ -f "$save_root/status/$run_id.done" ]] || return 1
        [[ ! -f "$save_root/status/$run_id.failed" ]] || return 1
        [[ ! -f "$save_root/status/$run_id.running" ]] || return 1
    done < "$manifest"
    [[ "$expected" -gt 0 ]]
}

run_until_complete() {
    local label="$1"
    local data_root="$2"
    local dataset_name="$3"
    local control_root="$4"
    local save_root="$5"
    local swanlab_group="$6"
    local attempt
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
        echo "PIPELINE dataset=$label attempt=$attempt/$MAX_ATTEMPTS started_at=$(date --iso-8601=seconds)"
        set +e
        DATA_ROOT="$data_root" \
        DATASET_NAME="$dataset_name" \
        CONTROL_ROOT="$control_root" \
        MANIFEST="$control_root/manifest.tsv" \
        SAVE_ROOT="$save_root" \
        SWANLAB_GROUP="$swanlab_group" \
        GPU_IDS="$GPU_IDS" \
        PYTHON_BIN="$PYTHON_BIN" \
        WAIT_FOR_LOCK=1 \
            bash "$TOOLS_DIR/run_nudt_mirsdt_all_models.sh"
        local exit_code=$?
        set -e
        if all_done "$control_root/manifest.tsv" "$save_root"; then
            echo "PIPELINE dataset=$label complete_at=$(date --iso-8601=seconds)"
            return 0
        fi
        echo "PIPELINE dataset=$label incomplete exit=$exit_code"
    done
    echo "PIPELINE_ABORT dataset=$label attempts=$MAX_ATTEMPTS" >&2
    return 1
}

summarize_and_analyze() {
    local label="$1"
    local control_root="$2"
    local save_root="$3"
    shift 3
    "$PYTHON_BIN" "$TOOLS_DIR/summarize_nudt_mirsdt_results.py" \
        --control-root "$control_root" \
        --save-root "$save_root" \
        --dataset-label "$label"
    "$PYTHON_BIN" "$TOOLS_DIR/analyze_nudt_experiments.py" \
        --results "$control_root/results.csv" \
        --dataset-label "$label" \
        --output "$control_root/ANALYSIS.md" \
        "$@"
}

echo "PIPELINE_START $(date --iso-8601=seconds)"
run_until_complete \
    "NUDT-MIRSDT" "$CLEAN_DATA" "NUDT-MIRSDT" \
    "$CLEAN_CONTROL" "$CLEAN_SAVE" "all-models-scratch-seed49"
summarize_and_analyze "NUDT-MIRSDT" "$CLEAN_CONTROL" "$CLEAN_SAVE"

run_until_complete \
    "NUDT-MIRSDT-Noise8.0_FJY" "$NOISE_DATA" "NUDT-MIRSDT-Noise8.0_FJY" \
    "$NOISE_CONTROL" "$NOISE_SAVE" "noise8-fjy-all-models-scratch-seed49"
summarize_and_analyze \
    "NUDT-MIRSDT-Noise8.0_FJY" "$NOISE_CONTROL" "$NOISE_SAVE" \
    --reference-results "$CLEAN_CONTROL/results.csv" \
    --reference-label "NUDT-MIRSDT clean"
touch "$NOISE_SAVE/PIPELINE_COMPLETE"
echo "PIPELINE_COMPLETE $(date --iso-8601=seconds)"
