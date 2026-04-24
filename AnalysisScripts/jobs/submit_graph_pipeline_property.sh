#!/bin/bash
#SBATCH --job-name=graphpipe_prop
#SBATCH --account=abucsek98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts/jobs/logs/graphpipe_property_%x_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -eo pipefail

SCRIPT_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts"
TMP_DIR="${SCRIPT_DIR}/jobs/tmp"
PIPE_OUT="${GRAPHPIPE_OUT_PATH:-/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisResults/PeriodicBoudaries/GraphPipeline}"
PROPERTY="${GRAPHPIPE_PROPERTY:?Set GRAPHPIPE_PROPERTY to topology, loop, pair_edge, node_connectivity, curvature, or nfd}"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
set -u

export GRAPHPIPE_PROJECT_ROOT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim"
export GRAPHPIPE_OUT_PATH="${PIPE_OUT}"
export GRAPHGEN_NODE_CONN_N_JOBS="${GRAPHGEN_NODE_CONN_N_JOBS:-36}"
export GRAPHGEN_NODE_CONN_VERBOSE="${GRAPHGEN_NODE_CONN_VERBOSE:-5}"
export GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS="${GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS:-36}"
export GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE="${GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE:-24}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMP_DIR}/mpl_${SLURM_JOB_ID}"
export JOBLIB_TEMP_FOLDER="${TMP_DIR}/joblib_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${JOBLIB_TEMP_FOLDER}" "${GRAPHPIPE_OUT_PATH}"

cd "${SCRIPT_DIR}"
python GraphPipelineComputeProperty.py --group "${PROPERTY}"
