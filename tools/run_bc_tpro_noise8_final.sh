#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
EXPECTED_DATA_ROOT="/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY"
DATA_ROOT="${DATA_ROOT:-$EXPECTED_DATA_ROOT}"
DATASET_NAME="NUDT-MIRSDT-Noise8.0_FJY"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
PROFILE="upstream8fa1a68_fp32"
EXPECTED_STAGE1_ROOT="$REPO_ROOT/experiments/bc_tpro_stage1_noise8_upstream_2026-09-09"
STAGE1_ROOT="${STAGE1_ROOT:-$EXPECTED_STAGE1_ROOT}"
LOCK_FILE="${LOCK_FILE:-$STAGE1_ROOT/LOCKED_CANDIDATE.json}"
FINAL_ROOT="${FINAL_ROOT:-$REPO_ROOT/experiments/bc_tpro_final_noise8_2026-09-09}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/log}"
LOG_ROOT="$SAVE_ROOT/sem_seg"
QUEUE_ROOT="${QUEUE_ROOT:-$LOG_ROOT/_queues/bc_tpro_final_noise8_2026-09-09}"
STATUS_ROOT="$QUEUE_ROOT/status"
METRICS_ROOT="$FINAL_ROOT/official_test_metrics"
DRY_RUN="${DRY_RUN:-0}"

export DATA_ROOT PYTHON_BIN REPO_ROOT
# shellcheck source=project_runtime_env.sh
source "$TOOLS_DIR/project_runtime_env.sh"
csig_require_allowed_gpus "0,1,2"

fail() {
    echo "$*" >&2
    exit 1
}

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
[[ "$DATA_ROOT" == "$EXPECTED_DATA_ROOT" ]] || \
    fail "Final protocol requires exact data root: $EXPECTED_DATA_ROOT"
[[ "$STAGE1_ROOT" == "$EXPECTED_STAGE1_ROOT" ]] || \
    fail "Final protocol requires the upstream-aligned Stage-1 root."
[[ "$LOCK_FILE" == "$EXPECTED_STAGE1_ROOT/LOCKED_CANDIDATE.json" ]] || \
    fail "Final protocol requires the upstream schema-2 candidate lock."
[[ "$SAVE_ROOT" == "$REPO_ROOT/log" ]] || \
    fail "Final protocol requires the repository log root: $REPO_ROOT/log"
[[ "$FINAL_ROOT" == "$REPO_ROOT/experiments/bc_tpro_final_noise8_2026-09-09" ]] || \
    fail "Final protocol requires the canonical final experiment root."
[[ "$QUEUE_ROOT" == "$LOG_ROOT/_queues/bc_tpro_final_noise8_2026-09-09" ]] || \
    fail "Final protocol requires the canonical queue root."

# A dry run only replays the candidate selector. A formal run first acquires
# the launcher lock, then atomically publishes (or exactly verifies) the
# immutable final protocol lock before constructing any training command.
plan_mode="plan"
plan_extra=()
if [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$STATUS_ROOT" "$QUEUE_ROOT/launcher_logs" "$METRICS_ROOT"
    exec 9>"$QUEUE_ROOT/.launch.lock"
    flock -n 9 || fail "Another final Noise8 launcher holds the lock."
    plan_mode="freeze"
    plan_extra=(--final-root "$FINAL_ROOT")
fi
PLAN_TSV="$(
    "$PYTHON_BIN" "$TOOLS_DIR/validate_bc_tpro_noise8_final.py" "$plan_mode" \
        --data-root "$DATA_ROOT" \
        --stage1-root "$STAGE1_ROOT" \
        --log-root "$LOG_ROOT" \
        --lock "$LOCK_FILE" \
        --profile "$PROFILE" \
        "${plan_extra[@]}" \
        --emit-tsv
)"
[[ -n "$PLAN_TSV" ]] || fail "Locked final plan contains no runs."
mapfile -t PLAN_ROWS <<<"$PLAN_TSV"
[[ "${#PLAN_ROWS[@]}" -eq 3 || "${#PLAN_ROWS[@]}" -eq 6 ]] || \
    fail "Locked final plan must contain 3 or 6 runs."

selected_variant="none"
for row in "${PLAN_ROWS[@]}"; do
    IFS=$'\t' read -r _run_id _wave _role _code _model variant \
        _seed _gpu _log_dir <<<"$row"
    if [[ "$variant" != "none" ]]; then
        selected_variant="$variant"
    fi
done

verify_frozen_plan() {
    local replayed
    replayed="$(
        "$PYTHON_BIN" "$TOOLS_DIR/validate_bc_tpro_noise8_final.py" \
            verify-frozen \
            --data-root "$DATA_ROOT" \
            --stage1-root "$STAGE1_ROOT" \
            --log-root "$LOG_ROOT" \
            --lock "$LOCK_FILE" \
            --profile "$PROFILE" \
            --final-root "$FINAL_ROOT" \
            --emit-tsv
    )"
    [[ "$replayed" == "$PLAN_TSV" ]] || \
        fail "Frozen final plan changed before official-test access."
}

checkpoint_command() {
    local run_id="$1" wave="$2" role="$3" code="$4" model="$5"
    local variant="$6" seed="$7" gpu="$8" log_dir="$9"
    local run_dir="$LOG_ROOT/$log_dir"
    local checkpoint="$run_dir/checkpoints/epoch_32_model.pth"
    local training_log="$run_dir/logs/$model.txt"
    "$PYTHON_BIN" "$TOOLS_DIR/validate_bc_tpro_noise8_final.py" checkpoint \
        --checkpoint "$checkpoint" \
        --training-log "$training_log" \
        --log-root "$LOG_ROOT" \
        --selected-variant "$selected_variant" \
        --profile "$PROFILE" \
        --run-id "$run_id" --wave "$wave" --role "$role" --code "$code" \
        --model "$model" --variant "$variant" --seed "$seed" --gpu "$gpu" \
        --log-dir "$log_dir"
}

run_training() {
    local row="$1"
    local run_id wave role code model variant seed gpu log_dir
    IFS=$'\t' read -r run_id wave role code model variant seed gpu log_dir <<<"$row"
    csig_require_allowed_gpu "$gpu"
    local run_dir="$LOG_ROOT/$log_dir"
    local done_file="$STATUS_ROOT/$run_id.train.done"
    local running_file="$STATUS_ROOT/$run_id.train.running"
    local failed_file="$STATUS_ROOT/$run_id.train.failed"
    local command=(
        "$PYTHON_BIN" -u "$REPO_ROOT/train.py"
        --model "$model"
        --structure_variant "$variant"
        --structure_bottleneck_channels 8
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
        --datapath "$DATA_ROOT"
        --dataset "$DATASET_NAME"
        --log_dir "$log_dir"
        --savepath "$SAVE_ROOT"
        --seqlen 40
        --patch_size 128
        --sample_rate 0.1
        --sequence_augmentation 0
        --loss soft_iou
        --threshold_eval 0.5
        --train_amp 0
        --eval_amp 0
        --upstream_compat 1
        --eval_chunk_rows 32
        --eval_interval 8
        --skip_inprocess_validation 1
        --early_stopping_patience 0
        --early_stopping_metric eval_iou
        --train_workers 4
        --val_workers 0
        --prefetch_factor 2
        --seed "$seed"
        --deterministic 1
        --resume never
        --run_test_after_train 0
        --use_swanlab 1
        --swanlab_project DeepPro-BC-TPro
        --swanlab_group bc-tpro-final80-noise8-upstream8fa1a68-fp32-scratch-locked
        --swanlab_mode cloud
        --swanlab_resume never
        --base_ckpt ""
        --spatial_ckpt ""
        --st_ckpt ""
        --freeze_pretrained 0
    )

    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'TRAIN role=%s wave=%s' "$role" "$wave"
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi
    if [[ -f "$done_file" ]]; then
        checkpoint_command "$run_id" "$wave" "$role" "$code" "$model" \
            "$variant" "$seed" "$gpu" "$log_dir"
        echo "SKIP validated training run=$run_id"
        return 0
    fi
    [[ ! -e "$running_file" && ! -e "$failed_file" ]] || \
        fail "Training status requires manual inspection: $run_id"
    if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -print -quit)" ]]; then
        fail "Non-empty final run directory has no completed receipt: $run_dir"
    fi

    printf 'run_id=%s\nphase=train\nstarted_at=%s\n' \
        "$run_id" "$(date --iso-8601=seconds)" > "$running_file"
    set +e
    "${command[@]}"
    local exit_code=$?
    set -e
    if [[ "$exit_code" -eq 0 ]]; then
        set +e
        checkpoint_command "$run_id" "$wave" "$role" "$code" "$model" \
            "$variant" "$seed" "$gpu" "$log_dir"
        exit_code=$?
        set -e
    fi
    if [[ "$exit_code" -ne 0 ]]; then
        printf 'failed_at=%s\nexit_code=%s\n' \
            "$(date --iso-8601=seconds)" "$exit_code" >> "$running_file"
        mv "$running_file" "$failed_file"
        return "$exit_code"
    fi
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$running_file"
    mv "$running_file" "$done_file"
    echo "DONE training run=$run_id"
}

verify_all_training() {
    local row run_id wave role code model variant seed gpu log_dir
    for row in "${PLAN_ROWS[@]}"; do
        IFS=$'\t' read -r run_id wave role code model variant seed gpu log_dir <<<"$row"
        [[ -f "$STATUS_ROOT/$run_id.train.done" ]] || \
            fail "Official test remains sealed; training is incomplete: $run_id"
        checkpoint_command "$run_id" "$wave" "$role" "$code" "$model" \
            "$variant" "$seed" "$gpu" "$log_dir"
    done
}

run_evaluation() {
    local row="$1"
    local run_id wave role code model variant seed gpu log_dir
    IFS=$'\t' read -r run_id wave role code model variant seed gpu log_dir <<<"$row"
    if [[ "$DRY_RUN" == 0 ]]; then
        verify_frozen_plan
    fi
    csig_require_allowed_gpu "$gpu"
    local run_dir="$LOG_ROOT/$log_dir"
    local metrics="$METRICS_ROOT/${run_id}__official_test.json"
    local eval_log="$run_dir/eval_epoch-32.txt"
    local attempted_file="$STATUS_ROOT/$run_id.test.attempted"
    local done_file="$STATUS_ROOT/$run_id.test.done"
    local failed_file="$STATUS_ROOT/$run_id.test.failed"
    local command=(
        "$PYTHON_BIN" -u "$REPO_ROOT/test.py"
        --epoch 32
        --gpu "$gpu"
        --seqlen 40
        --datapath "$DATA_ROOT"
        --dataset "$DATASET_NAME"
        --split test
        --logpath "$SAVE_ROOT"
        --log_dir "$log_dir"
        --test_workers 1
        --prefetch_factor 1
        --eval_chunk_rows 32
        --threshold_eval 0.5
        --threshold_grid_step 0
        --metrics_json "$metrics"
    )

    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'TEST role=%s after=all-training' "$role"
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi
    if [[ -f "$done_file" ]]; then
        "$PYTHON_BIN" "$TOOLS_DIR/analyze_bc_tpro_noise8_final.py" one \
            --data-root "$DATA_ROOT" --stage1-root "$STAGE1_ROOT" \
            --log-root "$LOG_ROOT" --lock "$LOCK_FILE" \
            --profile "$PROFILE" --final-root "$FINAL_ROOT" \
            --run-id "$run_id"
        echo "SKIP validated official test run=$run_id"
        return 0
    fi
    [[ ! -e "$attempted_file" && ! -e "$failed_file" ]] || \
        fail "Official test was already attempted; no automatic retry: $run_id"
    [[ ! -e "$metrics" && ! -e "$eval_log" ]] || \
        fail "Official-test output already exists without a completed receipt: $run_id"

    printf 'run_id=%s\nphase=official_test\nstarted_at=%s\n' \
        "$run_id" "$(date --iso-8601=seconds)" > "$attempted_file"
    set +e
    "${command[@]}"
    local exit_code=$?
    set -e
    if [[ "$exit_code" -eq 0 ]]; then
        set +e
        "$PYTHON_BIN" "$TOOLS_DIR/analyze_bc_tpro_noise8_final.py" one \
            --data-root "$DATA_ROOT" --stage1-root "$STAGE1_ROOT" \
            --log-root "$LOG_ROOT" --lock "$LOCK_FILE" \
            --profile "$PROFILE" --final-root "$FINAL_ROOT" \
            --run-id "$run_id"
        exit_code=$?
        set -e
    fi
    if [[ "$exit_code" -ne 0 ]]; then
        printf 'failed_at=%s\nexit_code=%s\n' \
            "$(date --iso-8601=seconds)" "$exit_code" >> "$attempted_file"
        mv "$attempted_file" "$failed_file"
        return "$exit_code"
    fi
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$attempted_file"
    mv "$attempted_file" "$done_file"
    echo "DONE official test run=$run_id"
}

echo "Locked Noise8 final protocol: selected=$selected_variant runs=${#PLAN_ROWS[@]}"
for wave in 1 2; do
    wave_rows=()
    for row in "${PLAN_ROWS[@]}"; do
        IFS=$'\t' read -r _run_id row_wave _rest <<<"$row"
        if [[ "$row_wave" -eq "$wave" ]]; then
            wave_rows+=("$row")
        fi
    done
    [[ "${#wave_rows[@]}" -gt 0 ]] || continue
    if [[ "$DRY_RUN" == 1 ]]; then
        for row in "${wave_rows[@]}"; do
            run_training "$row"
        done
    else
        pids=()
        for row in "${wave_rows[@]}"; do
            run_id="${row%%$'\t'*}"
            run_training "$row" > "$QUEUE_ROOT/launcher_logs/$run_id.train.log" 2>&1 &
            pids+=("$!")
        done
        failures=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failures=$((failures + 1))
            fi
        done
        [[ "$failures" -eq 0 ]] || \
            fail "Training wave $wave failed; official test remains sealed."
    fi
done

if [[ "$DRY_RUN" == 1 ]]; then
    echo "BARRIER official-test commands below are plans only; all training must finish first."
    for row in "${PLAN_ROWS[@]}"; do
        run_evaluation "$row"
    done
    exit 0
fi

verify_all_training
verify_frozen_plan
# Deliberately serial: each fixed checkpoint receives one isolated official-test
# process only after the full training barrier has passed.
for row in "${PLAN_ROWS[@]}"; do
    run_evaluation "$row"
done

"$PYTHON_BIN" "$TOOLS_DIR/analyze_bc_tpro_noise8_final.py" all \
    --data-root "$DATA_ROOT" --stage1-root "$STAGE1_ROOT" \
    --log-root "$LOG_ROOT" --lock "$LOCK_FILE" \
    --profile "$PROFILE" --final-root "$FINAL_ROOT" \
    --queue-root "$QUEUE_ROOT"
echo "Final Noise8 80/20 evaluation complete: $FINAL_ROOT"
