#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_CONFIG="${HERE}/config_final_load_state.yaml"
exec "${HERE}/submit_jobs.sh" "${1:---dry-run}"
