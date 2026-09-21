#!/bin/bash
#SBATCH --job-name=crystal_property
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/crystal_%x_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
PROPERTY="${CRYSTAL_PROPERTY:?Set CRYSTAL_PROPERTY to topology, loop, pair_edge, node_connectivity, curvature, or nfd}"
export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
export GRAPHGEN_OUT_PATH="${PROJECT}/AnalysisResults/crystal_references/0_graph_and_basic_stats/artifacts/pipeline"
export GRAPHGEN_NODE_CONN_N_JOBS="${SLURM_CPUS_PER_TASK}"
export GRAPHGEN_PAIR_EDGE_EXPORT_N_JOBS="${SLURM_CPUS_PER_TASK}"
export GRAPHGEN_PAIR_EDGE_EXPORT_CHUNK_SIZE=24
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/yfjin/mpl_crystal_${PROPERTY}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="/tmp/yfjin/cache_crystal_${PROPERTY}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export JOBLIB_TEMP_FOLDER="/tmp/yfjin/joblib_crystal_${PROPERTY}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${JOBLIB_TEMP_FOLDER}"
cd "${PROJECT}"
python AnalysisScripts/PostAnalysis/crystal_reference_networks.py \
  --config AnalysisScripts/PostAnalysis/crystal_reference_config.json \
  --stage property \
  --group "${PROPERTY}"
