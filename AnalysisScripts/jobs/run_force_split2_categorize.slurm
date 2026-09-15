#!/bin/bash
#SBATCH --job-name=force_split2_categorize
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/force_split2_categorize_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
PIPELINE="${PROJECT}/AnalysisScripts/PostAnalysis/PeriodicRubyPipeline"
CONFIGS=("${PIPELINE}/config_periodic_force_split2.yaml" "${PIPELINE}/config_final_load_force_split2.yaml")
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"

export PS1="${PS1:-}"
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
cd "${PIPELINE}"
python -c 'import sys; from pathlib import Path; from pipeline_common import load_config,ensure_layout; from merge_results import refresh_categorized_views; c=load_config(sys.argv[1]); refresh_categorized_views(ensure_layout(c),Path(c["analysis_root"]),Path(c["force_presentation_root"]))' "${CONFIG}"
