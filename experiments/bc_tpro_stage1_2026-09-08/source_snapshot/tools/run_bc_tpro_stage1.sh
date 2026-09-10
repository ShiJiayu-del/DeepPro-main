#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
CLEAN_DATA="${CLEAN_DATA:-$REPO_ROOT/../datasets/NUDT-MIRSDT}"
NOISE_DATA="${NOISE_DATA:-$REPO_ROOT/../datasets/NUDT-MIRSDT-Noise8.0_FJY}"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_ROOT/experiments/bc_tpro_stage1_2026-09-08}"
MANIFEST="${MANIFEST:-$EXPERIMENT_ROOT/manifest.tsv}"
TRAIN_LIST="${TRAIN_LIST:-$EXPERIMENT_ROOT/splits/train_sequences.txt}"
VAL_LIST="${VAL_LIST:-$EXPERIMENT_ROOT/splits/val_sequences.txt}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/log}"
QUEUE_ROOT="${QUEUE_ROOT:-$SAVE_ROOT/sem_seg/_queues/bc_tpro_stage1_2026-09-08}"
STATUS_ROOT="$QUEUE_ROOT/status"
METRICS_ROOT="$EXPERIMENT_ROOT/metrics"
DRY_RUN="${DRY_RUN:-0}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
EXPECTED_SOURCE_SHA="31e71368775adbc6c893bcb29ae1f39df6ffc83a693f42f84a4322bcf5235a8c"
EXPECTED_TRAIN_SHA="3e18da9a5c8dccea57b5155c244de1327ded05c6867f0059336f7a231ebbd967"
EXPECTED_VAL_SHA="bb92ecfdb0c379acc9971eaffed99154a2f985b063fd1f091aa8bceb09b07338"

export DATA_ROOT="$CLEAN_DATA" PYTHON_BIN REPO_ROOT
# shellcheck source=project_runtime_env.sh
source "$TOOLS_DIR/project_runtime_env.sh"
csig_require_allowed_gpus "0,1,2"

fail() {
    echo "$*" >&2
    exit 1
}

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ "$USE_SWANLAB" == 0 || "$USE_SWANLAB" == 1 ]] || fail "USE_SWANLAB must be 0 or 1."
[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
[[ -f "$MANIFEST" ]] || fail "Missing manifest: $MANIFEST"
for data_root in "$CLEAN_DATA" "$NOISE_DATA"; do
    [[ -d "$data_root" ]] || fail "Missing dataset: $data_root"
    [[ -f "$data_root/train.txt" ]] || fail "Missing $data_root/train.txt"
    [[ -f "$data_root/test.txt" ]] || fail "Missing $data_root/test.txt"
done
[[ -f "$TRAIN_LIST" ]] || fail "Missing fixed train split: $TRAIN_LIST"
[[ -f "$VAL_LIST" ]] || fail "Missing fixed validation split: $VAL_LIST"

actual_source_sha="$(sha256sum "$CLEAN_DATA/train.txt" | awk '{print $1}')"
actual_train_sha="$(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
actual_val_sha="$(sha256sum "$VAL_LIST" | awk '{print $1}')"
[[ "$actual_source_sha" == "$EXPECTED_SOURCE_SHA" ]] || \
    fail "Source train.txt changed: $actual_source_sha"
[[ "$actual_train_sha" == "$EXPECTED_TRAIN_SHA" ]] || \
    fail "Fixed train split changed: $actual_train_sha"
[[ "$actual_val_sha" == "$EXPECTED_VAL_SHA" ]] || \
    fail "Fixed validation split changed: $actual_val_sha"
cmp -s "$CLEAN_DATA/train.txt" "$NOISE_DATA/train.txt" || \
    fail "Clean and Noise8 train.txt files differ."

mkdir -p "$STATUS_ROOT" "$QUEUE_ROOT/launcher_logs" "$METRICS_ROOT"
if [[ "$DRY_RUN" == 0 ]]; then
    exec 9>"$QUEUE_ROOT/.launch.lock"
    flock -n 9 || fail "Another BC-TPro stage-1 launcher holds the lock."
fi

run_evaluation() {
    local run_id="$1"
    local gpu="$2"
    local log_dir="$3"
    local condition="$4"
    local dataset_name="$5"
    local data_root="$6"
    local metrics_path="$METRICS_ROOT/${run_id}__${condition}.json"
    local command=(
        "$PYTHON_BIN" -u "$REPO_ROOT/test.py"
        --epoch 32
        --gpu "$gpu"
        --seqlen 40
        --datapath "$data_root"
        --dataset "$dataset_name"
        --sequence_list "$VAL_LIST"
        --logpath "$SAVE_ROOT"
        --log_dir "$log_dir"
        --test_workers 1
        --prefetch_factor 1
        --eval_chunk_rows 32
        --amp
        --threshold_eval 0.5
        --threshold_grid_step 0.01
        --metrics_json "$metrics_path"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'EVAL'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        # A fresh CUDA process avoids an Xid 31 seen when the original
        # difference-convolution kernels switch from long-lived patch training
        # to full-frame inference. Serialize full-frame evaluations on this
        # shared three-GPU host as an additional safety boundary.
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

    if [[ -f "$done_file" ]]; then
        echo "SKIP completed run=$run_id"
        return 0
    fi
    if [[ -f "$failed_file" ]]; then
        echo "Previous failure requires inspection; refusing automatic retry: $failed_file" >&2
        return 1
    fi
    if [[ -d "$experiment_dir" ]] && \
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
        --datapath "$CLEAN_DATA"
        --dataset NUDT-MIRSDT
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
        --train_amp 1
        --eval_amp 1
        --eval_chunk_rows 32
        --eval_interval 8
        --skip_inprocess_validation 1
        --early_stopping_patience 0
        --train_workers 4
        --val_workers 1
        --prefetch_factor 2
        --seed "$seed"
        --deterministic 1
        --resume never
        --run_test_after_train 0
        --use_swanlab "$USE_SWANLAB"
        --swanlab_project DeepPro-BC-TPro
        --swanlab_group bc-tpro-stage1-scratch
        --swanlab_mode "$SWANLAB_MODE"
        --swanlab_resume never
        --base_ckpt ""
        --spatial_ckpt ""
        --st_ckpt ""
        --freeze_pretrained 0
    )

    printf 'RUN wave=%s gpu=%s seed=%s id=%s variant=%s\n' \
        "$wave" "$gpu" "$seed" "$run_id" "$variant"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'TRAIN'
        printf ' %q' "${command[@]}"
        printf '\n'
        run_evaluation "$run_id" "$gpu" "$log_dir" clean_val \
            NUDT-MIRSDT "$CLEAN_DATA"
        run_evaluation "$run_id" "$gpu" "$log_dir" noise8_val \
            NUDT-MIRSDT-Noise8.0_FJY "$NOISE_DATA"
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
        run_evaluation "$run_id" "$gpu" "$log_dir" clean_val \
            NUDT-MIRSDT "$CLEAN_DATA"
        exit_code=$?
    fi
    if [[ "$exit_code" -eq 0 ]]; then
        run_evaluation "$run_id" "$gpu" "$log_dir" noise8_val \
            NUDT-MIRSDT-Noise8.0_FJY "$NOISE_DATA"
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

echo "BC-TPro stage1: 12 scratch runs, GPUs=0,1,2, fixed split SHA=$EXPECTED_VAL_SHA"
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

if [[ "$DRY_RUN" == 0 ]]; then
    "$PYTHON_BIN" "$TOOLS_DIR/analyze_bc_tpro_stage1.py" \
        --experiment-root "$EXPERIMENT_ROOT"
fi
echo "BC-TPro stage1 complete."
