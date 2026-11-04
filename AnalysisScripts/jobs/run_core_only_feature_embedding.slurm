#!/bin/bash
#SBATCH --job-name=core_only_embedding
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/core_only_embedding_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

# Step 06 of the core-only force-cluster analysis (PCA / t-SNE embeddings and
# cross-validated separability). Needs scikit-learn, hence gnn_granular.
set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
export MPLCONFIGDIR="/tmp/yfjin/matplotlib_embedding_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT}/AnalysisScripts/PostAnalysis/ForceClusterCoreOnly"
/home/yfjin/Research/anaconda3/envs/gnn_granular/bin/python 06_feature_embedding.py --n-jobs "${SLURM_CPUS_PER_TASK}"
