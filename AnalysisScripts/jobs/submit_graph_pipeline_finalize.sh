#!/bin/bash
#SBATCH --job-name=graphpipe_final
#SBATCH --account=abucsek98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts/jobs/logs/graphpipe_finalize_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -eo pipefail

SCRIPT_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts"
TMP_DIR="${SCRIPT_DIR}/jobs/tmp"
PIPE_OUT="${GRAPHPIPE_OUT_PATH:-/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisResults/PeriodicBoudaries/GraphPipeline}"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
set -u

export GRAPHPIPE_PROJECT_ROOT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim"
export GRAPHPIPE_OUT_PATH="${PIPE_OUT}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMP_DIR}/mpl_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${GRAPHPIPE_OUT_PATH}"

cd "${SCRIPT_DIR}"
python GraphPipelineFinalize.py
