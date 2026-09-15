#!/bin/bash
#SBATCH --job-name=force_split2_build
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/force_split2_build_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
INPUTS=(
  "AnalysisResults/periodic_boundaries/2026-08-03/0_graph_and_basic_stats/graph_data"
  "AnalysisResults/final_load/0_graph_and_basic_stats/graph_data"
)
OUTPUTS=(
  "AnalysisResults/periodic_boundaries/2026-08-03/0_graph_and_basic_stats/force_splits/force_split2"
  "AnalysisResults/final_load/0_graph_and_basic_stats/force_splits/force_split2"
)
export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
cd "${PROJECT}"
python AnalysisScripts/PostAnalysis/relabel_particle_contact_threshold.py \
  --input-dir "${INPUTS[${SLURM_ARRAY_TASK_ID}]}" \
  --output-dir "${OUTPUTS[${SLURM_ARRAY_TASK_ID}]}"
