#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
CLEAN_CONTROL="$REPO_ROOT/experiments/nudt_mirsdt_all_models_2026-09-01"
CLEAN_SAVE="$REPO_ROOT/log/sem_seg/_queues/nudt_mirsdt_all_models_2026-09-01"
NOISE_CONTROL="$REPO_ROOT/experiments/nudt_mirsdt_noise8_fjy_all_models_2026-09-03"
NOISE_SAVE="$REPO_ROOT/log/sem_seg/_queues/nudt_mirsdt_noise8_fjy_all_models_2026-09-03"
PIPELINE_ROOT="$REPO_ROOT/log/sem_seg/_pipeline"
PIPELINE_LOG="$PIPELINE_ROOT/paper_metrics_after_noise8_2026-09-04.log"

mkdir -p "$PIPELINE_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
exec 8>"$PIPELINE_ROOT/.paper_metrics_after_noise8.lock"
flock -n 8 || { echo "Paper-metric pipeline is already running." >&2; exit 1; }

# Wait for the current Noise8 training launcher without polling its progress.
exec 9>"$NOISE_SAVE/.launch.lock"
echo "WAITING_FOR_NOISE_TRAINING $(date --iso-8601=seconds)"
flock 9
echo "NOISE_TRAINING_LOCK_RELEASED $(date --iso-8601=seconds)"

DATA_ROOT="$REPO_ROOT/../datasets_v1/NUDT-MIRSDT" \
DATASET_NAME=NUDT-MIRSDT \
CONTROL_ROOT="$CLEAN_CONTROL" \
SAVE_ROOT="$CLEAN_SAVE" \
PAPER_SCENARIO=clean \
    bash "$TOOLS_DIR/run_nudt_paper_metrics.sh"

DATA_ROOT="$REPO_ROOT/../datasets_v1/NUDT-MIRSDT-Noise8.0_FJY" \
DATASET_NAME=NUDT-MIRSDT-Noise8.0_FJY \
CONTROL_ROOT="$NOISE_CONTROL" \
SAVE_ROOT="$NOISE_SAVE" \
PAPER_SCENARIO=noise8 \
    bash "$TOOLS_DIR/run_nudt_paper_metrics.sh"

echo "PAPER_METRICS_PIPELINE_COMPLETE $(date --iso-8601=seconds)"
