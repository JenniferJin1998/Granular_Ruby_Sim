#!/bin/bash
#SBATCH --job-name=crystal_build
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12gb
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/crystal_build_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/yfjin/mpl_crystal_build_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/yfjin/cache_crystal_build_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
cd "${PROJECT}"
python AnalysisScripts/PostAnalysis/crystal_reference_networks.py \
  --config AnalysisScripts/PostAnalysis/crystal_reference_config.json \
  --stage build
