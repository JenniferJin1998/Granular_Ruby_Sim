#!/bin/bash
#SBATCH --job-name=force_split_boundaries
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --array=0-10%11
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/force_split_boundaries_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
DATASETS=(periodic_force_split2 periodic_force_split2 periodic_force_split2 final_load_force_split2 final_load_force_split2 final_load_force_split2 final_load_force_split2 jamming_final_threshold jamming_final_threshold jamming_final_threshold jamming_final_threshold)
REFERENCES=(all top bottom all top_bottom top bottom all top_bottom top bottom)
export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
export MPLCONFIGDIR="/tmp/yfjin/matplotlib_force_boundary_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT}"
python AnalysisScripts/PostAnalysis/analyze_boundary_layers.py \
  --datasets "${DATASETS[${SLURM_ARRAY_TASK_ID}]}" \
  --boundary-references "${REFERENCES[${SLURM_ARRAY_TASK_ID}]}" \
  --skip-3d
