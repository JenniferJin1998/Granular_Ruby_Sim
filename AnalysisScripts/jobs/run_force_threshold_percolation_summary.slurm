#!/bin/bash
#SBATCH --job-name=force_percolation_summary
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12gb
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/force_percolation_summary_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
export MPLCONFIGDIR="/tmp/yfjin/matplotlib_force_percolation_summary_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/yfjin/cache_force_percolation_summary_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${XDG_CACHE_HOME}"
cd "${PROJECT}"
python AnalysisScripts/PostAnalysis/force_threshold_percolation.py \
  --config AnalysisScripts/PostAnalysis/force_threshold_percolation_config.json \
  --summary
