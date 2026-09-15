#!/bin/bash
#SBATCH --job-name=force_split2_finalize
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/force_split2_finalize_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
PIPELINE="${PROJECT}/AnalysisScripts/PostAnalysis/PeriodicRubyPipeline"
CONFIGS=("${PIPELINE}/config_periodic_force_split2.yaml" "${PIPELINE}/config_final_load_force_split2.yaml")
OLD_JOB4=(
  "${PROJECT}/AnalysisResults/periodic_boundaries/2026-08-03/1_network_property_comparison/artifacts/local_structure_pipeline/job4_bond_order"
  "${PROJECT}/AnalysisResults/final_load/1_network_property_comparison/artifacts/local_structure_pipeline/job4_bond_order"
)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
export MPLCONFIGDIR="/tmp/yfjin/matplotlib_force_split2_finalize_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
cd "${PIPELINE}"

OUTPUT_ROOT=$(python -c 'import sys; from pipeline_common import load_config,ensure_layout; print(ensure_layout(load_config(sys.argv[1])))' "${CONFIG}")
cp -as "${OLD_JOB4[${SLURM_ARRAY_TASK_ID}]}/." "${OUTPUT_ROOT}/job4_bond_order/"
python merge_results.py --config "${CONFIG}" --allow-incomplete
python analyze_bivariate_relationships.py --config "${CONFIG}"
python -c 'import sys; from pathlib import Path; from pipeline_common import load_config,ensure_layout; from merge_results import refresh_categorized_views; c=load_config(sys.argv[1]); refresh_categorized_views(ensure_layout(c),Path(c["analysis_root"]),Path(c["force_presentation_root"]))' "${CONFIG}"
