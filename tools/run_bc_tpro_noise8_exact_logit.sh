#!/usr/bin/env bash
set -euo pipefail

# Serialized raw-logit sensitivity evaluation for Noise8-trained epoch-32 runs.
# This launcher never reads the official test split and never starts training.
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-modernized}"
case "$PROTOCOL_PROFILE" in
    modernized)
        DEFAULT_EXPERIMENT_ROOT="$REPO_ROOT/experiments/bc_tpro_stage1_noise8_2026-09-09"
        QUEUE_NAME="bc_tpro_stage1_noise8_2026-09-09"
        EXPECTED_USE_AMP=1
        PROTOCOL_CONFIG_ARGS=()
        ;;
    upstream8fa1a68_fp32)
        DEFAULT_EXPERIMENT_ROOT="$REPO_ROOT/experiments/bc_tpro_stage1_noise8_upstream_2026-09-09"
        QUEUE_NAME="bc_tpro_stage1_noise8_upstream_2026-09-09"
        EXPECTED_USE_AMP=0
        PROTOCOL_CONFIG_ARGS=()
        ;;
    *)
        echo "Unknown PROTOCOL_PROFILE: $PROTOCOL_PROFILE" >&2
        exit 1
        ;;
esac
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$DEFAULT_EXPERIMENT_ROOT}"
if [[ "$PROTOCOL_PROFILE" == upstream8fa1a68_fp32 ]]; then
    [[ "$(realpath -m -- "$EXPERIMENT_ROOT")" == "$DEFAULT_EXPERIMENT_ROOT" ]] || {
        echo "Upstream exact-logit experiment root is fixed: $DEFAULT_EXPERIMENT_ROOT" >&2
        exit 1
    }
    PROTOCOL_CONFIG_ARGS=(
        --protocol-config "$EXPERIMENT_ROOT/UPSTREAM_PROTOCOL.json"
    )
fi
MANIFEST="${MANIFEST:-$EXPERIMENT_ROOT/manifest.tsv}"
TRAIN_LIST="${TRAIN_LIST:-$EXPERIMENT_ROOT/splits/train_sequences.txt}"
VAL_LIST="${VAL_LIST:-$EXPERIMENT_ROOT/splits/val_sequences.txt}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$EXPERIMENT_ROOT/splits/split_manifest.json}"
NOISE_DATA="${NOISE_DATA:-/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/log}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/exact_logit_metrics}"
QUEUE_ROOT="${QUEUE_ROOT:-$SAVE_ROOT/sem_seg/_queues/$QUEUE_NAME}"
STATUS_ROOT="$QUEUE_ROOT/status"
EVALUATION_LOCK="$QUEUE_ROOT/.evaluation.lock"
DRY_RUN="${DRY_RUN:-0}"
USE_AMP="${USE_AMP:-$EXPECTED_USE_AMP}"

export DATA_ROOT="$NOISE_DATA" PYTHON_BIN REPO_ROOT
# shellcheck source=project_runtime_env.sh
source "$TOOLS_DIR/project_runtime_env.sh"
csig_require_allowed_gpus "0,1,2"

fail() {
    echo "$*" >&2
    exit 1
}

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ "$USE_AMP" == "$EXPECTED_USE_AMP" ]] || \
    fail "Profile $PROTOCOL_PROFILE requires USE_AMP=$EXPECTED_USE_AMP."
[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
[[ -f "$MANIFEST" ]] || fail "Missing manifest: $MANIFEST"
[[ -f "$VAL_LIST" ]] || fail "Missing validation list: $VAL_LIST"
[[ -d "$NOISE_DATA" ]] || fail "Missing Noise8 dataset: $NOISE_DATA"

"$PYTHON_BIN" "$TOOLS_DIR/validate_bc_tpro_noise8_setup.py" \
    --data-root "$NOISE_DATA" \
    --expected-data-root "/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY" \
    --dataset NUDT-MIRSDT-Noise8.0_FJY \
    --manifest "$MANIFEST" \
    --train-list "$TRAIN_LIST" \
    --val-list "$VAL_LIST" \
    --split-manifest "$SPLIT_MANIFEST" \
    --profile "$PROTOCOL_PROFILE" \
    "${PROTOCOL_CONFIG_ARGS[@]}"

expected_val=$'Sequence9\nSequence13\nSequence14\nSequence16\nSequence17\nSequence20\nSequence29\nSequence31\nSequence45\nSequence49\nSequence55\nSequence61\nSequence68\nSequence74\nSequence77\nSequence84'
[[ "$(sed '/^[[:space:]]*$/d' "$VAL_LIST")" == "$expected_val" ]] || \
    fail "Validation list is not the frozen ordered 16-sequence split."
mapfile -t manifest_rows < <(awk -F $'\t' 'NR > 1 {print}' "$MANIFEST")
[[ "${#manifest_rows[@]}" -eq 12 ]] || fail "Expected exactly 12 manifest rows."

if [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$OUTPUT_ROOT" "$QUEUE_ROOT/exact_logit_logs" "$(dirname -- "$EVALUATION_LOCK")"
    exec 9>"$QUEUE_ROOT/.exact_logit_launch.lock"
    flock -n 9 || fail "Another Noise8 exact-logit launcher is active."
fi

check_completed() {
    local row="$1"
    local run_id _wave _model _variant _seed _gpu log_dir
    IFS=$'\t' read -r run_id _wave _model _variant _seed _gpu log_dir <<<"$row"
    local checkpoint="$SAVE_ROOT/sem_seg/$log_dir/checkpoints/epoch_32_model.pth"
    local grid_metric="$EXPERIMENT_ROOT/metrics/${run_id}__noise8_internal_val.json"
    local done_file="$STATUS_ROOT/$run_id.done"
    for artifact in "$checkpoint" "$grid_metric" "$done_file"; do
        if [[ ! -f "$artifact" ]]; then
            if [[ "$DRY_RUN" == 1 ]]; then
                echo "DRY_RUN pending completed-run artifact: $artifact" >&2
            else
                fail "All training and probability-grid evaluation must finish first: $artifact"
            fi
        fi
    done
}

for row in "${manifest_rows[@]}"; do
    check_completed "$row"
done

run_one() {
    local row="$1"
    local repeat_index="$2"
    local reference_json="${3:-}"
    local run_id _wave model variant seed gpu log_dir
    IFS=$'\t' read -r run_id _wave model variant seed gpu log_dir <<<"$row"
    [[ "$model" == DeepPro-Plus_BCTPro ]] || fail "Unexpected model in manifest: $model"
    csig_require_allowed_gpu "$gpu"

    local repeat_suffix=""
    if [[ "$repeat_index" -eq 1 ]]; then
        repeat_suffix="__repeat1"
    fi
    local stem="${run_id}__noise8_internal_val__exact_logit${repeat_suffix}"
    local output_json="$OUTPUT_ROOT/${stem}.json"
    local output_npz="$OUTPUT_ROOT/${stem}.npz"
    if [[ -f "$output_json" && -f "$output_npz" ]]; then
        echo "SKIP completed exact-logit output: $stem"
        return 0
    fi
    if [[ -e "$output_json" || -e "$output_npz" ]]; then
        fail "Partial output exists for $stem; inspect it before retrying."
    fi

    local command=(
        "$PYTHON_BIN" -u "$TOOLS_DIR/evaluate_bc_tpro_noise8_exact_logit.py"
        --gpu "$gpu"
        --datapath "$NOISE_DATA"
        --dataset NUDT-MIRSDT-Noise8.0_FJY
        --sequence-list "$VAL_LIST"
        --logpath "$SAVE_ROOT"
        --log-dir "$log_dir"
        --run-id "$run_id"
        --epoch 32
        --seqlen 40
        --seed "$seed"
        --structure-variant "$variant"
        --repeat-index "$repeat_index"
        --eval-chunk-rows 32
        --test-workers 1
        --prefetch-factor 1
        --low-fa-cap 5e-5
        --output-json "$output_json"
        --output-npz "$output_npz"
        --profile "$PROTOCOL_PROFILE"
    )
    if [[ "$USE_AMP" == 1 ]]; then
        command+=(--amp)
    fi
    if [[ -n "$reference_json" ]]; then
        [[ -f "$reference_json" || "$DRY_RUN" == 1 ]] || \
            fail "Missing same-seed B1 raw-logit reference: $reference_json"
        command+=(--reference-json "$reference_json")
    fi

    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'EXACT_LOGIT'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        (
            flock 8
            "${command[@]}"
        ) 8>"$EVALUATION_LOCK" \
            >"$QUEUE_ROOT/exact_logit_logs/${stem}.log" 2>&1
        echo "DONE exact-logit $stem"
    fi
}

# Primary B1 references are always created before any candidate evaluation.
for row in "${manifest_rows[@]}"; do
    IFS=$'\t' read -r _run_id _wave _model variant _seed _gpu _log_dir <<<"$row"
    [[ "$variant" == none ]] || continue
    run_one "$row" 0
done

# Primary C0, C1, and C2 evaluations use the same-seed B1 logit-zero budget.
for wanted_variant in temporal_control center_multiscale center_ring; do
    for row in "${manifest_rows[@]}"; do
        IFS=$'\t' read -r _run_id _wave _model variant seed _gpu _log_dir <<<"$row"
        [[ "$variant" == "$wanted_variant" ]] || continue
        reference="$OUTPUT_ROOT/b1_none_seed${seed}__noise8_internal_val__exact_logit.json"
        run_one "$row" 0 "$reference"
    done
done

# One independent repeat of every B1 and C2 run checks exact integer counts.
for wanted_variant in none center_ring; do
    for row in "${manifest_rows[@]}"; do
        IFS=$'\t' read -r _run_id _wave _model variant seed _gpu _log_dir <<<"$row"
        [[ "$variant" == "$wanted_variant" ]] || continue
        reference=""
        if [[ "$variant" == center_ring ]]; then
            reference="$OUTPUT_ROOT/b1_none_seed${seed}__noise8_internal_val__exact_logit.json"
        fi
        run_one "$row" 1 "$reference"
    done
done

analysis_command=(
    "$PYTHON_BIN" "$TOOLS_DIR/analyze_bc_tpro_noise8_exact_logit.py"
    --experiment-root "$EXPERIMENT_ROOT"
    --log-root "$SAVE_ROOT/sem_seg"
    --profile "$PROTOCOL_PROFILE"
)
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'ANALYZE'
    printf ' %q' "${analysis_command[@]}"
    printf '\n'
else
    "${analysis_command[@]}"
fi

echo "Noise8 exact-logit profile=$PROTOCOL_PROFILE evaluation complete: $OUTPUT_ROOT"
