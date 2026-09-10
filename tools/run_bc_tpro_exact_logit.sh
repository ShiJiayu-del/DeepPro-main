#!/usr/bin/env bash
set -euo pipefail

# Post-hoc protocol-amendment evaluator. It never changes the original
# 109-point probability-grid JSON files.
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/sjyPID/bin/python}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_ROOT/experiments/bc_tpro_stage1_2026-09-08}"
MANIFEST="${MANIFEST:-$EXPERIMENT_ROOT/manifest.tsv}"
VAL_LIST="${VAL_LIST:-$EXPERIMENT_ROOT/splits/val_sequences.txt}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/log}"
CLEAN_DATA="${CLEAN_DATA:-$REPO_ROOT/../datasets/NUDT-MIRSDT}"
NOISE_DATA="${NOISE_DATA:-$REPO_ROOT/../datasets/NUDT-MIRSDT-Noise8.0_FJY}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/exact_logit_metrics}"
QUEUE_ROOT="${QUEUE_ROOT:-$SAVE_ROOT/sem_seg/_queues/bc_tpro_stage1_exact_logit_2026-09-09}"
EVALUATION_LOCK="${EVALUATION_LOCK:-$SAVE_ROOT/sem_seg/_queues/bc_tpro_stage1_2026-09-08/.evaluation.lock}"
DRY_RUN="${DRY_RUN:-0}"
USE_AMP="${USE_AMP:-1}"
RETENTION_MODE="${RETENTION_MODE:-target-events}"

export DATA_ROOT="$CLEAN_DATA" PYTHON_BIN REPO_ROOT
# shellcheck source=project_runtime_env.sh
source "$TOOLS_DIR/project_runtime_env.sh"
csig_require_allowed_gpus "0,1,2"

fail() {
    echo "$*" >&2
    exit 1
}

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ "$USE_AMP" == 0 || "$USE_AMP" == 1 ]] || fail "USE_AMP must be 0 or 1."
[[ "$RETENTION_MODE" == target-events || "$RETENTION_MODE" == full-reference-grid ]] || \
    fail "RETENTION_MODE must be target-events or full-reference-grid."
[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
[[ -f "$MANIFEST" ]] || fail "Missing manifest: $MANIFEST"
[[ -f "$VAL_LIST" ]] || fail "Missing validation sequence list: $VAL_LIST"
[[ -d "$CLEAN_DATA" && -d "$NOISE_DATA" ]] || fail "Missing Clean or Noise8 dataset."

mkdir -p "$OUTPUT_ROOT" "$QUEUE_ROOT" "$(dirname -- "$EVALUATION_LOCK")"
if [[ "$DRY_RUN" == 0 ]]; then
    exec 9>"$QUEUE_ROOT/.launch.lock"
    flock -n 9 || fail "Another exact-logit launcher holds $QUEUE_ROOT/.launch.lock"
fi

run_one() {
    local row="$1"
    local condition="$2"
    local dataset_name="$3"
    local data_root="$4"
    local reference_json="${5:-}"
    local effective_retention_mode="$RETENTION_MODE"
    if [[ -z "$reference_json" ]]; then
        # A B1 output creates F_ref/D_ref and therefore has no prior reference.
        effective_retention_mode=target-events
    fi
    local run_id wave model variant seed gpu log_dir
    IFS=$'\t' read -r run_id wave model variant seed gpu log_dir <<<"$row"
    csig_require_allowed_gpu "$gpu"

    local stem="${run_id}__${condition}__exact_logit"
    local output_json="$OUTPUT_ROOT/${stem}.json"
    local output_npz="$OUTPUT_ROOT/${stem}.npz"
    local checkpoint="$SAVE_ROOT/sem_seg/$log_dir/checkpoints/epoch_32_model.pth"
    if [[ ! -f "$checkpoint" ]]; then
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "DRY_RUN pending checkpoint: $checkpoint" >&2
        else
            fail "Missing completed epoch-32 checkpoint: $checkpoint"
        fi
    fi
    if [[ -f "$output_json" && -f "$output_npz" ]]; then
        echo "SKIP exact-logit output already exists: $stem"
        return 0
    fi
    if [[ -e "$output_json" || -e "$output_npz" ]]; then
        fail "Partial output exists for $stem; inspect it before retrying."
    fi

    local command=(
        "$PYTHON_BIN" -u "$TOOLS_DIR/evaluate_bc_tpro_exact_logit.py"
        --gpu "$gpu"
        --datapath "$data_root"
        --dataset "$dataset_name"
        --sequence-list "$VAL_LIST"
        --condition "$condition"
        --logpath "$SAVE_ROOT"
        --log-dir "$log_dir"
        --epoch 32
        --seqlen 40
        --seed "$seed"
        --structure-variant "$variant"
        --eval-chunk-rows 32
        --test-workers 1
        --prefetch-factor 1
        --low-fa-cap 5e-5
        --reference-retention-mode "$effective_retention_mode"
        --output-json "$output_json"
        --output-npz "$output_npz"
    )
    if [[ "$USE_AMP" == 1 ]]; then
        command+=(--amp)
    fi
    if [[ -n "$reference_json" ]]; then
        [[ -f "$reference_json" || "$DRY_RUN" == 1 ]] || \
            fail "Missing B1 exact-logit reference: $reference_json"
        command+=(--reference-json "$reference_json")
    fi

    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'EXACT_LOGIT'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        # Full-frame evaluations are serialized because this host has a known
        # unsafe concurrent/deterministic cuDNN Conv3d path.
        (
            flock 8
            "${command[@]}"
        ) 8>"$EVALUATION_LOCK" \
            >"$QUEUE_ROOT/${stem}.log" 2>&1
        echo "DONE exact-logit $stem"
    fi
}

mapfile -t manifest_rows < <(awk -F $'\t' 'NR > 1 {print}' "$MANIFEST")
[[ "${#manifest_rows[@]}" -eq 12 ]] || fail "Expected the frozen 12-run manifest."

# Phase 1 produces each seed/condition B1 workpoint at raw logit zero.
for row in "${manifest_rows[@]}"; do
    IFS=$'\t' read -r run_id _wave _model variant _seed _gpu _log_dir <<<"$row"
    [[ "$variant" == none ]] || continue
    run_one "$row" clean_val NUDT-MIRSDT "$CLEAN_DATA"
    run_one "$row" noise8_val NUDT-MIRSDT-Noise8.0_FJY "$NOISE_DATA"
done

# Phase 2 evaluates every candidate against the same-seed/same-condition B1.
for row in "${manifest_rows[@]}"; do
    IFS=$'\t' read -r _run_id _wave _model variant seed _gpu _log_dir <<<"$row"
    [[ "$variant" != none ]] || continue
    clean_reference="$OUTPUT_ROOT/b1_none_seed${seed}__clean_val__exact_logit.json"
    noise_reference="$OUTPUT_ROOT/b1_none_seed${seed}__noise8_val__exact_logit.json"
    run_one "$row" clean_val NUDT-MIRSDT "$CLEAN_DATA" "$clean_reference"
    run_one "$row" noise8_val NUDT-MIRSDT-Noise8.0_FJY "$NOISE_DATA" "$noise_reference"
done

echo "Exact-logit protocol-amendment evaluation complete: $OUTPUT_ROOT"
