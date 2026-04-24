#!/bin/bash
#SBATCH --job-name=graphgen_pbc_1sim
#SBATCH --account=abucsek98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts/jobs/logs/graphgen_pbc_1sim_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -eo pipefail

SCRIPT_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts"
TMP_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts/jobs/tmp"
OUTPUT_ROOT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisResults/PeriodicBoudaries/GraphGenerationTests/single_sim_all_cpus"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export GRAPHGEN_GEOMETRY_FILTER="0deg"
export GRAPHGEN_MAX_SIMS_PER_GEOMETRY=1
export GRAPHGEN_RUN_SIM_PARALLEL=0
export GRAPHGEN_NODE_CONN_N_JOBS=36
export GRAPHGEN_NODE_CONN_VERBOSE=5
export GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS=36
export GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE=24
export GRAPHGEN_ENABLE_TIMING_LOGS=1
export GRAPHGEN_OUT_PATH="${RUN_OUTPUT_DIR}"

export MPLCONFIGDIR="${TMP_DIR}/mpl_${SLURM_JOB_ID}"
export JOBLIB_TEMP_FOLDER="${TMP_DIR}/joblib_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${JOBLIB_TEMP_FOLDER}" "${RUN_OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

echo "========================================="
echo "Single-simulation graph timing job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "GRAPHGEN_GEOMETRY_FILTER=${GRAPHGEN_GEOMETRY_FILTER}"
echo "GRAPHGEN_MAX_SIMS_PER_GEOMETRY=${GRAPHGEN_MAX_SIMS_PER_GEOMETRY}"
echo "GRAPHGEN_RUN_SIM_PARALLEL=${GRAPHGEN_RUN_SIM_PARALLEL}"
echo "GRAPHGEN_NODE_CONN_N_JOBS=${GRAPHGEN_NODE_CONN_N_JOBS}"
echo "GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS=${GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS}"
echo "GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE=${GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE}"
echo "GRAPHGEN_OUT_PATH=${GRAPHGEN_OUT_PATH}"
echo "========================================="

python GraphGeneration.py

echo "========================================="
echo "Single-simulation graph timing job finished: $(date)"
echo "========================================="
