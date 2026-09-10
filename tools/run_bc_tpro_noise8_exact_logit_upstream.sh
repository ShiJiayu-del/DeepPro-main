#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
EXPECTED_EXPERIMENT_ROOT="$REPO_ROOT/experiments/bc_tpro_stage1_noise8_upstream_2026-09-09"

if [[ -n "${EXPERIMENT_ROOT:-}" ]] && \
   [[ "$(realpath -m -- "$EXPERIMENT_ROOT")" != "$EXPECTED_EXPERIMENT_ROOT" ]]; then
    echo "Upstream exact-logit experiment root is fixed: $EXPECTED_EXPERIMENT_ROOT" >&2
    exit 1
fi

export PROTOCOL_PROFILE=upstream8fa1a68_fp32
export EXPERIMENT_ROOT="$EXPECTED_EXPERIMENT_ROOT"
export USE_AMP=0
export QUEUE_ROOT="${QUEUE_ROOT:-$REPO_ROOT/log/sem_seg/_queues/bc_tpro_stage1_noise8_upstream_2026-09-09}"

exec "$TOOLS_DIR/run_bc_tpro_noise8_exact_logit.sh" "$@"
