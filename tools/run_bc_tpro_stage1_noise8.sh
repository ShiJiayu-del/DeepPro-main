#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
EXPECTED_NOISE_DATA_ROOT="/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY"
NOISE_DATA="${NOISE_DATA:-$EXPECTED_NOISE_DATA_ROOT}"
DATASET_NAME="NUDT-MIRSDT-Noise8.0_FJY"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_ROOT/experiments/bc_tpro_stage1_noise8_2026-09-09}"
MANIFEST="$EXPERIMENT_ROOT/manifest.tsv"
TRAIN_LIST="$EXPERIMENT_ROOT/splits/train_sequences.txt"
VAL_LIST="$EXPERIMENT_ROOT/splits/val_sequences.txt"
SPLIT_MANIFEST="$EXPERIMENT_ROOT/splits/split_manifest.json"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/log}"
QUEUE_ROOT="${QUEUE_ROOT:-$SAVE_ROOT/sem_seg/_queues/bc_tpro_stage1_noise8_2026-09-09}"
STATUS_ROOT="$QUEUE_ROOT/status"
METRICS_ROOT="$EXPERIMENT_ROOT/metrics"
DRY_RUN="${DRY_RUN:-0}"
PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-modernized}"

case "$PROTOCOL_PROFILE" in
    modernized)
        TRAIN_AMP=1
        EVAL_AMP=1
        SWANLAB_GROUP="bc-tpro-stage1-noise8-scratch"
        TRAIN_PROFILE_ARGS=()
        EVAL_PROFILE_ARGS=(--amp)
        VALIDATOR_PROFILE_ARGS=(--profile modernized)
        ;;
    upstream8fa1a68_fp32)
        TRAIN_AMP=0
        EVAL_AMP=0
        SWANLAB_GROUP="bc-tpro-stage1-noise8-upstream8fa1a68-fp32-scratch"
        TRAIN_PROFILE_ARGS=(--upstream_compat 1)
        EVAL_PROFILE_ARGS=()
        VALIDATOR_PROFILE_ARGS=(
            --profile upstream8fa1a68_fp32
            --protocol-config "$EXPERIMENT_ROOT/UPSTREAM_PROTOCOL.json"
        )
        ;;
    *)
        echo "Unknown PROTOCOL_PROFILE: $PROTOCOL_PROFILE" >&2
        exit 1
        ;;
esac

export DATA_ROOT="$NOISE_DATA" PYTHON_BIN REPO_ROOT
# shellcheck source=project_runtime_env.sh
source "$TOOLS_DIR/project_runtime_env.sh"
csig_require_allowed_gpus "0,1,2"

fail() {
    echo "$*" >&2
    exit 1
}

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"

"$PYTHON_BIN" "$TOOLS_DIR/validate_bc_tpro_noise8_setup.py" \
    --data-root "$NOISE_DATA" \
    --expected-data-root "$EXPECTED_NOISE_DATA_ROOT" \
    --dataset "$DATASET_NAME" \
    --manifest "$MANIFEST" \
    --train-list "$TRAIN_LIST" \
    --val-list "$VAL_LIST" \
    --split-manifest "$SPLIT_MANIFEST" \
    "${VALIDATOR_PROFILE_ARGS[@]}"

if [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$STATUS_ROOT" "$QUEUE_ROOT/launcher_logs" "$METRICS_ROOT"
    exec 9>"$QUEUE_ROOT/.launch.lock"
    flock -n 9 || fail "Another Noise8 BC-TPro stage-1 launcher holds the lock."
fi

run_evaluation() {
    local run_id="$1"
    local gpu="$2"
    local log_dir="$3"
    local metrics_path="$METRICS_ROOT/${run_id}__noise8_internal_val.json"
    local command=(
        "$PYTHON_BIN" -u "$REPO_ROOT/test.py"
        --epoch 32
        --gpu "$gpu"
        --seqlen 40
        --datapath "$NOISE_DATA"
        --dataset "$DATASET_NAME"
        --sequence_list "$VAL_LIST"
        --logpath "$SAVE_ROOT"
        --log_dir "$log_dir"
        --test_workers 1
        --prefetch_factor 1
        --eval_chunk_rows 32
        --threshold_eval 0.5
        --threshold_grid_step 0.01
        --metrics_json "$metrics_path"
        "${EVAL_PROFILE_ARGS[@]}"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'EVAL'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        (
            flock 8
            "${command[@]}"
        ) 8>"$QUEUE_ROOT/.evaluation.lock"
    fi
}

run_job() {
    local row="$1"
    local run_id wave model variant seed gpu log_dir
    IFS=$'\t' read -r run_id wave model variant seed gpu log_dir <<<"$row"
    csig_require_allowed_gpu "$gpu"

    local done_file="$STATUS_ROOT/$run_id.done"
    local running_file="$STATUS_ROOT/$run_id.running"
    local failed_file="$STATUS_ROOT/$run_id.failed"
    local experiment_dir="$SAVE_ROOT/sem_seg/$log_dir"

    if [[ "$DRY_RUN" == 0 && -f "$done_file" ]]; then
        echo "SKIP completed run=$run_id"
        return 0
    fi
    if [[ "$DRY_RUN" == 0 && -f "$failed_file" ]]; then
        echo "Previous failure requires inspection; refusing automatic retry: $failed_file" >&2
        return 1
    fi
    if [[ "$DRY_RUN" == 0 && -d "$experiment_dir" ]] && \
       [[ -n "$(find "$experiment_dir" -mindepth 1 -print -quit)" ]]; then
        echo "Experiment directory is non-empty without .done; refusing overwrite: $experiment_dir" >&2
        return 1
    fi

    local command=(
        "$PYTHON_BIN" -u "$REPO_ROOT/train.py"
        --model "$model"
        --structure_variant "$variant"
        --batch_size 4
        --gradient_accumulation_steps 1
        --epoch 32
        --learning_rate 0.001
        --optimizer Adam
        --decay_rate 0.0001
        --step_size 10
        --lr_decay 0.7
        --gpu "$gpu"
        --gpu_num 1
        --datapath "$NOISE_DATA"
        --dataset "$DATASET_NAME"
        --train_sequence_list "$TRAIN_LIST"
        --val_sequence_list "$VAL_LIST"
        --log_dir "$log_dir"
        --savepath "$SAVE_ROOT"
        --seqlen 40
        --patch_size 128
        --sample_rate 0.1
        --sequence_augmentation 0
        --loss soft_iou
        --threshold_eval 0.5
        --train_amp "$TRAIN_AMP"
        --eval_amp "$EVAL_AMP"
        --eval_chunk_rows 32
        --eval_interval 8
        --skip_inprocess_validation 1
        --early_stopping_patience 0
        --early_stopping_metric eval_iou
        --train_workers 4
        --val_workers 1
        --prefetch_factor 2
        --seed "$seed"
        --deterministic 1
        --resume never
        --run_test_after_train 0
        --use_swanlab 1
        --swanlab_project DeepPro-BC-TPro
        --swanlab_group "$SWANLAB_GROUP"
        --swanlab_mode cloud
        --swanlab_resume never
        --base_ckpt ""
        --spatial_ckpt ""
        --st_ckpt ""
        --freeze_pretrained 0
        "${TRAIN_PROFILE_ARGS[@]}"
    )

    printf 'RUN wave=%s gpu=%s seed=%s id=%s variant=%s\n' \
        "$wave" "$gpu" "$seed" "$run_id" "$variant"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'TRAIN'
        printf ' %q' "${command[@]}"
        printf '\n'
        run_evaluation "$run_id" "$gpu" "$log_dir"
        return 0
    fi

    printf 'run_id=%s\nwave=%s\ngpu=%s\nseed=%s\nstarted_at=%s\n' \
        "$run_id" "$wave" "$gpu" "$seed" "$(date --iso-8601=seconds)" \
        > "$running_file"
    mark_interrupted() {
        local signal_exit=$?
        if [[ -f "$running_file" ]]; then
            printf 'interrupted_at=%s\nexit_code=%s\n' \
                "$(date --iso-8601=seconds)" "$signal_exit" \
                >> "$running_file"
            mv "$running_file" "$failed_file"
        fi
    }
    trap mark_interrupted HUP INT TERM EXIT
    local started_seconds=$SECONDS
    set +e
    "${command[@]}"
    local exit_code=$?
    if [[ "$exit_code" -eq 0 ]]; then
        run_evaluation "$run_id" "$gpu" "$log_dir"
        exit_code=$?
    fi
    set -e

    local elapsed_seconds=$((SECONDS - started_seconds))
    if [[ "$exit_code" -eq 0 ]]; then
        printf 'finished_at=%s\nelapsed_seconds=%s\n' \
            "$(date --iso-8601=seconds)" "$elapsed_seconds" >> "$running_file"
        mv "$running_file" "$done_file"
        trap - HUP INT TERM EXIT
        echo "DONE run=$run_id seconds=$elapsed_seconds"
        return 0
    fi
    printf 'failed_at=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
        "$(date --iso-8601=seconds)" "$elapsed_seconds" "$exit_code" \
        >> "$running_file"
    mv "$running_file" "$failed_file"
    trap - HUP INT TERM EXIT
    echo "FAILED run=$run_id exit=$exit_code" >&2
    return "$exit_code"
}

echo "Noise8 BC-TPro stage1 profile=$PROTOCOL_PROFILE: 12 scratch runs, GPUs=0,1,2, split=64/16"
for wave in 1 2 3 4; do
    mapfile -t wave_jobs < <(
        awk -F $'\t' -v target="$wave" 'NR > 1 && $2 == target {print}' "$MANIFEST"
    )
    [[ "${#wave_jobs[@]}" -eq 3 ]] || fail "Wave $wave must contain exactly 3 jobs."
    echo "START wave=$wave jobs=${#wave_jobs[@]}"
    wave_pids=()
    for row in "${wave_jobs[@]}"; do
        run_id="${row%%$'\t'*}"
        if [[ "$DRY_RUN" == 1 ]]; then
            run_job "$row"
        else
            run_job "$row" > "$QUEUE_ROOT/launcher_logs/$run_id.log" 2>&1 &
            wave_pids+=("$!")
        fi
    done
    if [[ "$DRY_RUN" == 0 ]]; then
        wave_failures=0
        for pid in "${wave_pids[@]}"; do
            if ! wait "$pid"; then
                wave_failures=$((wave_failures + 1))
            fi
        done
        [[ "$wave_failures" -eq 0 ]] || \
            fail "Wave $wave has $wave_failures failed job(s); later waves were not started."
    fi
    echo "DONE wave=$wave"
done

echo "Noise8 BC-TPro stage1 complete; metrics: $METRICS_ROOT"
