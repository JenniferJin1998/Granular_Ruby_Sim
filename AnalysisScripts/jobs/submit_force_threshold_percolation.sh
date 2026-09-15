#!/bin/bash
set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
cd "${PROJECT}"
worker_job=$(sbatch AnalysisScripts/jobs/run_force_threshold_percolation.slurm | awk '{print $4}')
summary_job=$(sbatch --dependency="afterok:${worker_job}" AnalysisScripts/jobs/run_force_threshold_percolation_summary.slurm | awk '{print $4}')
printf 'worker_job=%s\nsummary_job=%s\n' "${worker_job}" "${summary_job}"
