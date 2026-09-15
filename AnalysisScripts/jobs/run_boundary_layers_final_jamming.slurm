#!/bin/bash
#SBATCH --job-name=boundary_layers
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/boundary_layers_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail

PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
DATASETS=(final_load final_load final_load final_load jamming jamming jamming jamming)
REFERENCES=(all top_bottom top bottom all top_bottom top bottom)
DATASET="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"
REFERENCE="${REFERENCES[${SLURM_ARRAY_TASK_ID}]}"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
export MPLCONFIGDIR="/tmp/yfjin/matplotlib_boundary_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd "${PROJECT}"
echo "dataset=${DATASET} boundary_reference=${REFERENCE}"
python AnalysisScripts/PostAnalysis/analyze_boundary_layers.py \
  --datasets "${DATASET}" \
  --boundary-references "${REFERENCE}" \
  --skip-3d
