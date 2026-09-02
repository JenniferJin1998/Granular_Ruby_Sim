#!/bin/bash
#SBATCH --job-name=gp_prod_final
#SBATCH --account=abucsek98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts/jobs/logs/gp_prod_finalize_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -eo pipefail

SCRIPT_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts"
PIPELINE_DIR="${SCRIPT_DIR}/Pipeline"
TMP_DIR="${SCRIPT_DIR}/jobs/tmp"
PIPE_OUT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisResults/PeriodicBoudaries/GraphPipeline_0deg_30deg"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
set -u

export GRAPHPIPE_PROJECT_ROOT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim"
export GRAPHPIPE_OUT_PATH="${PIPE_OUT}"
export GRAPHGEN_GEOMETRY_FILTER="0deg,30deg"
export GRAPHGEN_MAX_SIMS_PER_GEOMETRY=0
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMP_DIR}/mpl_${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${GRAPHPIPE_OUT_PATH}" "${SCRIPT_DIR}/jobs/logs"

cd "${PIPELINE_DIR}"
echo "Production pipeline finalize started: $(date)"
echo "GRAPHPIPE_OUT_PATH=${GRAPHPIPE_OUT_PATH}"
python GraphPipelineFinalize.py
echo "Production pipeline finalize finished: $(date)"
