#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/../datasets_v1/NUDT-MIRSDT}"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
MANIFEST="${MANIFEST:-$REPO_ROOT/experiments/nudt_mirsdt_all_models_2026-09-01/manifest.tsv}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/log/nudt_mirsdt_all_models_2026-09-01}"
STATUS_ROOT="$SAVE_ROOT/status"
GPU_IDS="${GPU_IDS:-0,1,2}"
EPOCHS="${EPOCHS:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2}"
SEED="${SEED:-49}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
DRY_RUN="${DRY_RUN:-0}"

export DATA_ROOT PYTHON_BIN REPO_ROOT
# shellcheck source=project_runtime_env.sh
source "$TOOLS_DIR/project_runtime_env.sh"
csig_require_allowed_gpus "$GPU_IDS"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Experiment manifest does not exist: $MANIFEST" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
for required in train.txt test.txt; do
    if [[ ! -f "$DATA_ROOT/$required" ]]; then
        echo "NUDT-MIRSDT split file is missing: $DATA_ROOT/$required" >&2
        exit 1
    fi
done
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi

IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
if [[ "${#gpu_array[@]}" -ne 3 ]]; then
    echo "This comparison requires exactly three allowed GPUs, got: $GPU_IDS" >&2
    exit 1
fi

mkdir -p "$STATUS_ROOT" "$SAVE_ROOT/launcher_logs"
if [[ "$DRY_RUN" == 0 ]]; then
    exec 9>"$SAVE_ROOT/.launch.lock"
    if ! flock -n 9; then
        echo "Another NUDT-MIRSDT all-model launcher already holds the lock." >&2
        exit 1
    fi
fi
mapfile -t jobs < <(tail -n +2 "$MANIFEST")
if [[ "${#jobs[@]}" -eq 0 ]]; then
    echo "Manifest contains no jobs: $MANIFEST" >&2
    exit 1
fi

run_job() {
    local gpu_id="$1"
    local job="$2"
    local run_id model variant loss
    IFS=$'\t' read -r run_id model variant loss <<<"$job"
    local done_file="$STATUS_ROOT/$run_id.done"
    local failed_file="$STATUS_ROOT/$run_id.failed"
    local running_file="$STATUS_ROOT/$run_id.running"
    local launcher_log="$SAVE_ROOT/launcher_logs/$run_id.log"
    local experiment_dir="$SAVE_ROOT/sem_seg/$run_id"
    local resume_mode="never"

    if [[ -f "$done_file" ]]; then
        echo "SKIP completed run=$run_id gpu=$gpu_id"
        return 0
    fi
    if [[ -f "$experiment_dir/checkpoints/latest_model.pth" ]]; then
        resume_mode="auto"
    fi

    local command=(
        "$PYTHON_BIN" -u "$REPO_ROOT/train.py"
        --model "$model"
        --batch_size "$BATCH_SIZE"
        --epoch "$EPOCHS"
        --learning_rate 0.005
        --gpu "$gpu_id"
        --gpu_num 1
        --datapath "$DATA_ROOT"
        --dataset NUDT-MIRSDT
        --log_dir "$run_id"
        --savepath "$SAVE_ROOT"
        --seqlen 40
        --patch_size 128
        --sample_rate 0.1
        --loss "$loss"
        --threshold_eval 0.5
        --train_amp 1
        --eval_amp 1
        --eval_interval "$EVAL_INTERVAL"
        --train_workers 4
        --val_workers 2
        --use_swanlab "$USE_SWANLAB"
        --swanlab_project DeepPro-NUDT-MIRSDT
        --swanlab_group all-models-scratch-seed49
        --swanlab_mode "$SWANLAB_MODE"
        --swanlab_resume never
        --seed "$SEED"
        --resume "$resume_mode"
        --run_test_after_train 0
        --base_ckpt ""
        --spatial_ckpt ""
        --st_ckpt ""
        --freeze_pretrained 0
    )
    if [[ "$variant" != "-" ]]; then
        command+=(--structure_variant "$variant")
    fi

    printf 'RUN gpu=%s id=%s model=%s variant=%s loss=%s resume=%s\n' \
        "$gpu_id" "$run_id" "$model" "$variant" "$loss" "$resume_mode"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'COMMAND'
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi

    rm -f "$failed_file"
    printf 'run_id=%s\ngpu=%s\nstarted_at=%s\n' \
        "$run_id" "$gpu_id" "$(date --iso-8601=seconds)" > "$running_file"
    local start_seconds=$SECONDS
    set +e
    "${command[@]}" 2>&1 | tee "$launcher_log"
    local exit_code=${PIPESTATUS[0]}
    set -e
    local elapsed_seconds=$((SECONDS - start_seconds))
    rm -f "$running_file"
    if [[ "$exit_code" -eq 0 ]]; then
        printf 'run_id=%s\ngpu=%s\nfinished_at=%s\nelapsed_seconds=%s\n' \
            "$run_id" "$gpu_id" "$(date --iso-8601=seconds)" \
            "$elapsed_seconds" > "$done_file"
        "$PYTHON_BIN" "$TOOLS_DIR/summarize_nudt_mirsdt_results.py" \
            --quiet || true
        echo "DONE run=$run_id gpu=$gpu_id seconds=$elapsed_seconds"
        return 0
    fi

    printf 'run_id=%s\ngpu=%s\nfailed_at=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
        "$run_id" "$gpu_id" "$(date --iso-8601=seconds)" \
        "$elapsed_seconds" "$exit_code" > "$failed_file"
    echo "FAILED run=$run_id gpu=$gpu_id exit=$exit_code" >&2
    return "$exit_code"
}

run_queue() {
    local queue_index="$1"
    local gpu_id="${gpu_array[$queue_index]}"
    local failures=0
    local index
    for index in "${!jobs[@]}"; do
        if (( index % ${#gpu_array[@]} != queue_index )); then
            continue
        fi
        if ! run_job "$gpu_id" "${jobs[$index]}"; then
            failures=$((failures + 1))
        fi
    done
    return "$failures"
}

echo "NUDT-MIRSDT jobs=${#jobs[@]} GPUs=$GPU_IDS epochs=$EPOCHS batch=$BATCH_SIZE"
if [[ "$DRY_RUN" == 1 ]]; then
    for queue_index in "${!gpu_array[@]}"; do
        run_queue "$queue_index"
    done
    exit 0
fi

launcher_pids=()
for queue_index in "${!gpu_array[@]}"; do
    run_queue "$queue_index" > \
        "$SAVE_ROOT/launcher_logs/queue_gpu${gpu_array[$queue_index]}.log" 2>&1 &
    launcher_pids+=("$!")
done

failures=0
for pid in "${launcher_pids[@]}"; do
    if ! wait "$pid"; then
        failures=$((failures + 1))
    fi
done

"$PYTHON_BIN" "$TOOLS_DIR/summarize_nudt_mirsdt_results.py"
if [[ "$failures" -ne 0 ]]; then
    echo "$failures GPU queue(s) contained failed experiments." >&2
    exit 1
fi
echo "ALL_NUDT_MIRSDT_EXPERIMENTS_COMPLETE"
