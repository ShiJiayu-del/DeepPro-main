#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
DATASET_NAME="${DATASET_NAME:?DATASET_NAME is required}"
CONTROL_ROOT="${CONTROL_ROOT:?CONTROL_ROOT is required}"
SAVE_ROOT="${SAVE_ROOT:?SAVE_ROOT is required}"
PAPER_SCENARIO="${PAPER_SCENARIO:?PAPER_SCENARIO must be clean or noise8}"
MANIFEST="${MANIFEST:-$CONTROL_ROOT/manifest.tsv}"
GPU_IDS="${GPU_IDS:-0,1,2}"
EVAL_CHUNK_ROWS="${EVAL_CHUNK_ROWS:-32}"
STATUS_ROOT="$SAVE_ROOT/paper_metrics/status"
RAW_ROOT="$SAVE_ROOT/paper_metrics/raw"

# shellcheck source=project_runtime_env.sh
source "$TOOLS_DIR/project_runtime_env.sh"
csig_require_allowed_gpus "$GPU_IDS"
IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
if [[ "${#gpu_array[@]}" -ne 3 ]]; then
    echo "Paper evaluation requires exactly three GPUs." >&2
    exit 1
fi
for path in "$DATA_ROOT/train.txt" "$DATA_ROOT/test.txt" "$MANIFEST"; do
    [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

mkdir -p "$STATUS_ROOT" "$RAW_ROOT" "$SAVE_ROOT/paper_metrics/logs"
exec 9>"$SAVE_ROOT/paper_metrics/.evaluation.lock"
if ! flock -n 9; then
    echo "Paper-aligned evaluation is already running for $DATASET_NAME." >&2
    exit 1
fi
mapfile -t jobs < <(tail -n +2 "$MANIFEST")

run_job() {
    local gpu_id="$1"
    local job="$2"
    local run_id model variant loss log_dir
    IFS=$'\t' read -r run_id model variant loss log_dir <<<"$job"
    local model_save_root="$SAVE_ROOT"
    if [[ -n "$log_dir" ]]; then
        model_save_root="$REPO_ROOT/log"
    else
        log_dir="$run_id"
    fi
    local experiment="$model_save_root/sem_seg/$log_dir"
    local checkpoint="$experiment/checkpoints/best_model.pth"
    local metrics="$RAW_ROOT/$run_id.json"
    local done_file="$STATUS_ROOT/$run_id.done"
    local failed_file="$STATUS_ROOT/$run_id.failed"
    local running_file="$STATUS_ROOT/$run_id.running"
    local log_file="$SAVE_ROOT/paper_metrics/logs/$run_id.log"
    if [[ -f "$done_file" && -s "$metrics" ]]; then
        echo "SKIP paper metrics run=$run_id"
        return 0
    fi
    if [[ ! -f "$checkpoint" ]]; then
        printf 'run_id=%s\nreason=missing_best_checkpoint\n' "$run_id" > "$failed_file"
        echo "MISSING checkpoint run=$run_id" >&2
        return 1
    fi

    local command=(
        "$PYTHON_BIN" -u "$REPO_ROOT/test.py"
        --gpu "$gpu_id"
        --seqlen 40
        --datapath "$DATA_ROOT"
        --dataset "$DATASET_NAME"
        --logpath "$model_save_root"
        --log_dir "$log_dir"
        --threshold_eval 0.5
        --test_workers 2
        --prefetch_factor 1
        --metrics_json "$metrics"
    )
    # FeedbackSTS training/validation use FP32: its feedback products can
    # overflow in FP16, producing invalid probabilities and misleading counts.
    if [[ "$run_id" != "feedbacksts" ]]; then
        command+=(--amp)
    fi
    case "$model" in
        DeepPro-Plus|DeepPro-Plus_BRTD|DeepPro-Plus_BRTD2|DeepPro-Plus_BRTD3|DeepPro-Plus_BRTD3_PointCenter)
            command+=(--eval_chunk_rows "$EVAL_CHUNK_ROWS")
            ;;
    esac
    if [[ "$run_id" == "deeppro_plus_tdcsta" ]]; then
        command=(env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "${command[@]}")
    fi

    rm -f "$failed_file"
    printf 'run_id=%s\ngpu=%s\nstarted_at=%s\n' \
        "$run_id" "$gpu_id" "$(date --iso-8601=seconds)" > "$running_file"
    set +e
    "${command[@]}" > "$log_file" 2>&1
    local exit_code=$?
    set -e
    rm -f "$running_file"
    if [[ "$exit_code" -eq 0 && -s "$metrics" ]]; then
        printf 'run_id=%s\ngpu=%s\nfinished_at=%s\n' \
            "$run_id" "$gpu_id" "$(date --iso-8601=seconds)" > "$done_file"
        echo "DONE paper metrics run=$run_id gpu=$gpu_id"
        return 0
    fi
    printf 'run_id=%s\ngpu=%s\nexit_code=%s\nfailed_at=%s\n' \
        "$run_id" "$gpu_id" "$exit_code" "$(date --iso-8601=seconds)" > "$failed_file"
    echo "FAILED paper metrics run=$run_id gpu=$gpu_id exit=$exit_code" >&2
    return 1
}

run_queue() {
    local queue_index="$1"
    local failures=0
    local index
    for index in "${!jobs[@]}"; do
        if (( index % ${#gpu_array[@]} != queue_index )); then
            continue
        fi
        run_job "${gpu_array[$queue_index]}" "${jobs[$index]}" || failures=$((failures + 1))
    done
    return "$failures"
}

pids=()
for queue_index in "${!gpu_array[@]}"; do
    run_queue "$queue_index" > \
        "$SAVE_ROOT/paper_metrics/logs/queue_gpu${gpu_array[$queue_index]}.log" 2>&1 &
    pids+=("$!")
done
failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done

"$PYTHON_BIN" "$TOOLS_DIR/summarize_nudt_paper_metrics.py" \
    --control-root "$CONTROL_ROOT" \
    --save-root "$SAVE_ROOT" \
    --dataset-label "$DATASET_NAME" \
    --paper-scenario "$PAPER_SCENARIO"
if [[ "$failures" -ne 0 ]]; then
    echo "$failures evaluation queue(s) contained failures." >&2
    exit 1
fi
echo "ALL_PAPER_METRICS_COMPLETE dataset=$DATASET_NAME"
