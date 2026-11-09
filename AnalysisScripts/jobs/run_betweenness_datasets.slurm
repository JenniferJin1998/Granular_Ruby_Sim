#!/bin/bash
#SBATCH --job-name=betweenness_datasets
#SBATCH --account=abucsek0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/jobs/logs/betweenness_datasets_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yfjin@umich.edu

# Core-only betweenness steps for the periodic-boundary force clusters (01, 02, 05) and the
# final-load high-stress classification (01, 02, 05, 07, 09), run side by side.
# Labels must already exist (00_prepare_dataset_labels.py with BC_DATASET set).
set -euo pipefail
PROJECT="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608"
PY=/home/yfjin/Research/anaconda3/envs/graph_analysis/bin/python
export MPLCONFIGDIR="/tmp/yfjin/matplotlib_betweenness_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT}/AnalysisScripts/PostAnalysis/ForceClusterCoreOnly"

run_set() {  # dataset, steps...
    local dataset=$1; shift
    for step in "$@"; do
        echo "=== ${dataset}: ${step} ($(date +%T))"
        BC_DATASET=${dataset} ${PY} ${step}
    done
}

run_set periodic_force 01_betweenness_vs_hydrostatic_stress.py 02_global_betweenness_rank.py \
    05_correlation.py > "${PROJECT}/AnalysisScripts/jobs/logs/periodic_force_${SLURM_JOB_ID}.out" 2>&1 &
pid_periodic=$!
run_set final_load_stress 01_betweenness_vs_hydrostatic_stress.py 02_global_betweenness_rank.py \
    05_correlation.py 07_radial_control.py 09_radial_auc.py \
    > "${PROJECT}/AnalysisScripts/jobs/logs/final_load_stress_${SLURM_JOB_ID}.out" 2>&1 &
pid_stress=$!

status=0
wait ${pid_periodic} || { echo "periodic_force failed"; status=1; }
wait ${pid_stress} || { echo "final_load_stress failed"; status=1; }
BC_DATASET=final_load_stress ${PY} 09_radial_auc.py --bins 6 > /dev/null
exit ${status}
