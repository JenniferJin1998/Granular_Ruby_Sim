#!/bin/bash
#SBATCH --job-name=gp_prod_prop
#SBATCH --account=abucsek98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=18:00:00
#SBATCH --array=0-39%8
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts/jobs/logs/gp_prod_%x_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -eo pipefail

SCRIPT_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts"
PIPELINE_DIR="${SCRIPT_DIR}/Pipeline"
TMP_DIR="${SCRIPT_DIR}/jobs/tmp"
PIPE_OUT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisResults/PeriodicBoudaries/GraphPipeline_0deg_30deg"
PROPERTY="${GRAPHPIPE_PROPERTY:?Set GRAPHPIPE_PROPERTY to topology, loop, pair_edge, node_connectivity, curvature, or nfd}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 20 )); then
    GEOMETRY="0deg"
    SIM_IDX="${TASK_ID}"
else
    GEOMETRY="30deg"
    SIM_IDX="$((TASK_ID - 20))"
fi

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
set -u

export GRAPHPIPE_PROJECT_ROOT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim"
export GRAPHPIPE_OUT_PATH="${PIPE_OUT}"
export GRAPHGEN_GEOMETRY_FILTER="0deg,30deg"
export GRAPHGEN_MAX_SIMS_PER_GEOMETRY=0
export GRAPHGEN_NODE_CONN_N_JOBS="${GRAPHGEN_NODE_CONN_N_JOBS:-${SLURM_CPUS_PER_TASK}}"
export GRAPHGEN_NODE_CONN_VERBOSE="${GRAPHGEN_NODE_CONN_VERBOSE:-5}"
export GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS="${GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS:-${SLURM_CPUS_PER_TASK}}"
export GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE="${GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE:-24}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMP_DIR}/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export JOBLIB_TEMP_FOLDER="${TMP_DIR}/joblib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${MPLCONFIGDIR}" "${JOBLIB_TEMP_FOLDER}" "${GRAPHPIPE_OUT_PATH}" "${SCRIPT_DIR}/jobs/logs"

cd "${PIPELINE_DIR}"
echo "Production property job started: $(date)"
echo "Property=${PROPERTY}"
echo "Geometry=${GEOMETRY}"
echo "Simulation=${SIM_IDX}"
echo "CPUs=${SLURM_CPUS_PER_TASK}"
echo "GRAPHPIPE_OUT_PATH=${GRAPHPIPE_OUT_PATH}"
python GraphPipelineComputeProperty.py --group "${PROPERTY}" --geometry "${GEOMETRY}" --sim-idx "${SIM_IDX}"
echo "Production property job finished: $(date)"
