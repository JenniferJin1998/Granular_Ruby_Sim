#!/bin/bash
#SBATCH --job-name=crystal_finalize
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/crystal_finalize_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/yfjin/mpl_crystal_finalize_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/yfjin/cache_crystal_finalize_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
cd "${PROJECT}"
python AnalysisScripts/PostAnalysis/crystal_reference_networks.py \
  --config AnalysisScripts/PostAnalysis/crystal_reference_config.json \
  --stage finalize
