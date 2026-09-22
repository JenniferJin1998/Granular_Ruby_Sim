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

JOURNAL="${LOG_ROOT}/submission_$(date +%Y%m%d_%H%M%S).txt"
record_job() { printf '%s=%s\n' "$1" "$2" | tee -a "${JOURNAL}"; }

BUILD=$(sbatch --parsable AnalysisScripts/jobs/run_crystal_reference_build.slurm)
record_job build "${BUILD}"

TOPOLOGY=$(CRYSTAL_PROPERTY=topology sbatch --parsable \
  --dependency="afterok:${BUILD}" --job-name=crystal_topology \
  --array=0-3%4 --cpus-per-task=4 --mem=16gb --time=06:00:00 \
  AnalysisScripts/jobs/run_crystal_reference_property.slurm)
record_job topology "${TOPOLOGY}"

LOOP=$(CRYSTAL_PROPERTY=loop sbatch --parsable \
  --dependency="afterok:${BUILD}" --job-name=crystal_loop \
  --array=0-3%4 --cpus-per-task=1 --mem=32gb --time=7-00:00:00 \
  AnalysisScripts/jobs/run_crystal_reference_property.slurm)
record_job loop "${LOOP}"

PAIR=$(CRYSTAL_PROPERTY=pair_edge sbatch --parsable \
  --dependency="afterok:${BUILD}" --job-name=crystal_pair \
  --array=0-3%4 --cpus-per-task=16 --mem=48gb --time=12:00:00 \
  AnalysisScripts/jobs/run_crystal_reference_property.slurm)
record_job pair_edge "${PAIR}"

NODE=$(CRYSTAL_PROPERTY=node_connectivity sbatch --parsable \
  --dependency="afterok:${BUILD}" --job-name=crystal_nodeconn \
  --array=0-3%4 --cpus-per-task=16 --mem=48gb --time=18:00:00 \
  AnalysisScripts/jobs/run_crystal_reference_property.slurm)
record_job node_connectivity "${NODE}"

CURVATURE=$(CRYSTAL_PROPERTY=curvature sbatch --parsable \
  --dependency="afterok:${BUILD}" --job-name=crystal_curvature \
  --array=0-3%4 --cpus-per-task=4 --mem=16gb --time=06:00:00 \
  AnalysisScripts/jobs/run_crystal_reference_property.slurm)
record_job curvature "${CURVATURE}"

NFD=$(CRYSTAL_PROPERTY=nfd sbatch --parsable \
  --dependency="afterok:${BUILD}" --job-name=crystal_nfd \
  --array=0-3%4 --cpus-per-task=4 --mem=16gb --time=06:00:00 \
  AnalysisScripts/jobs/run_crystal_reference_property.slurm)
record_job nfd "${NFD}"

FINALIZE=$(sbatch --parsable \
  --dependency="afterok:${TOPOLOGY}:${LOOP}:${PAIR}:${NODE}:${CURVATURE}:${NFD}" \
  AnalysisScripts/jobs/run_crystal_reference_finalize.slurm)
record_job finalize "${FINALIZE}"

printf 'Submitted crystal references: build=%s topology=%s loop=%s pair=%s node=%s curvature=%s nfd=%s finalize=%s\n' \
  "${BUILD}" "${TOPOLOGY}" "${LOOP}" "${PAIR}" "${NODE}" "${CURVATURE}" "${NFD}" "${FINALIZE}"
