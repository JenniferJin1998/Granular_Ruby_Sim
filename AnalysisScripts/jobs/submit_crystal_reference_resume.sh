#!/bin/bash
set -euo pipefail

PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
LOG_ROOT="${PROJECT}/AnalysisResults/crystal_references/logs"
cd "${PROJECT}"
mkdir -p "${LOG_ROOT}" AnalysisScripts/jobs/logs

if ! timeout 15s scontrol ping >/dev/null 2>&1; then
  echo "Slurm controller is unavailable; no crystal-reference jobs were submitted." >&2
  exit 1
fi

JOURNAL="${LOG_ROOT}/resume_submission_$(date +%Y%m%d_%H%M%S).txt"
record_job() { printf '%s=%s\n' "$1" "$2" | tee -a "${JOURNAL}"; }

# SC (array index 0) already completed. Resume only BCC, FCC, and HCP. The
# NetworkX minimum-cycle-basis calculation is single-process and exceeded the
# original 20-hour limit, so these tasks receive a seven-day wall-time limit.
LOOP=$(CRYSTAL_PROPERTY=loop sbatch --parsable \
  --job-name=crystal_loop_resume \
  --array=1-3%3 --cpus-per-task=1 --mem=32gb --time=7-00:00:00 \
  AnalysisScripts/jobs/run_crystal_reference_property.slurm)
record_job loop_resume "${LOOP}"

FINALIZE=$(sbatch --parsable \
  --dependency="afterok:${LOOP}" \
  AnalysisScripts/jobs/run_crystal_reference_finalize.slurm)
record_job finalize "${FINALIZE}"

printf 'Submitted missing loop calculations and finalizer: loop=%s finalize=%s\n' \
  "${LOOP}" "${FINALIZE}"
