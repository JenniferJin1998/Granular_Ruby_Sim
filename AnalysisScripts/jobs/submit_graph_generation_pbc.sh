#!/bin/bash
#SBATCH --job-name=graph_generation_pbc
#SBATCH --account=abucsek98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128gb
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts/jobs/logs/graph_generation_pbc_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -eo pipefail

SCRIPT_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts"
TMP_DIR="${SCRIPT_DIR}/jobs/tmp"
OUTPUT_ROOT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisResults/PeriodicBoudaries/GraphGeneration"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
set -u

# Keep threaded math libraries from oversubscribing the SLURM allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# There are 20 simulations per geometry, so a 20-worker pool fills the outer parallelism cleanly.
export GRAPHGEN_SIM_N_JOBS=20
export GRAPHGEN_NODE_CONN_N_JOBS_WHEN_SIM_PARALLEL=1
export PYTHONUNBUFFERED=1
export GRAPHGEN_OUT_PATH="${RUN_OUTPUT_DIR}"

# Use writable runtime directories under scratch so matplotlib/joblib do not fall back unexpectedly.
export MPLCONFIGDIR="${TMP_DIR}/mpl_${SLURM_JOB_ID}"
export JOBLIB_TEMP_FOLDER="${TMP_DIR}/joblib_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${JOBLIB_TEMP_FOLDER}" "${RUN_OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

echo "========================================="
echo "Graph generation job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "GRAPHGEN_SIM_N_JOBS=${GRAPHGEN_SIM_N_JOBS}"
echo "GRAPHGEN_OUT_PATH=${GRAPHGEN_OUT_PATH}"
echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER}"
echo "========================================="

python GraphGeneration.py

echo "========================================="
echo "Graph generation job finished: $(date)"
echo "========================================="
