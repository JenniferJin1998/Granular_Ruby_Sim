#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${PIPELINE_CONFIG:-${HERE}/config.yaml}"
CONDA_SH="/home/yfjin/Research/anaconda3/etc/profile.d/conda.sh"
MODE="${1:---dry-run}"

export PS1="${PS1:-}"
source "${CONDA_SH}"
conda activate graph_analysis
export MPLCONFIGDIR="/tmp/yfjin/matplotlib-periodic-pipeline"
mkdir -p "${MPLCONFIGDIR}"

readarray -t SETTINGS < <(python - "${CONFIG}" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); s=c['slurm']
print(c['output_root'] if c['output_root'].startswith('/') else c['project_root']+'/'+c['output_root'])
print(s['account']);print(s['partition']);print(s['job2_cpus']);print(s['job2_memory']);print(s['job2_time']);print(s['job2_array_concurrency'])
print(s['job3_cpus']);print(s['job3_memory']);print(s['job3_time']);print(s['job4_cpus']);print(s['job4_memory']);print(s['job4_time']);print(s['job4_array_concurrency'])
print(len(c['angles'])*c['simulations_per_angle']*len(c['hop_sizes'])*len(c['job2_property_bundles']))
print(s['job5_cpus']);print(s['job5_memory']);print(s['job5_time']);print(c['job5_array_concurrency'])
print(s['job6_cpus']);print(s['job6_memory']);print(s['job6_time']);print(c['job6_array_concurrency'])
nsim=len(c['angles'])*c['simulations_per_angle'];print(nsim);print(nsim*(1+len(c['hop_sizes'])))
root=c['project_root']
for key in ('feature_comparison_input','feature_comparison_output'):
 p=c.get(key,'');print(p if p.startswith('/') else root+'/'+p if p else '')
print(' '.join(c['angles']))
PY
)
OUT=${SETTINGS[0]}; ACCOUNT=${SETTINGS[1]}; PARTITION=${SETTINGS[2]}; J2N=${SETTINGS[14]}
J2BLOCK=$((J2N / 5))
NSIM=${SETTINGS[23]}; J5N=${SETTINGS[24]}
J4N=$((NSIM + 4))
FEATURE_INPUT=${SETTINGS[25]}; FEATURE_OUTPUT=${SETTINGS[26]}; GEOMETRIES=${SETTINGS[27]}
mkdir -p "${OUT}/logs"

run_local() {
  python "${HERE}/job0_verify_graphs.py" --config "${CONFIG}"
  python "${HERE}/job1_global_figures.py" --config "${CONFIG}"
  for ((i=0;i<J2N;i++)); do python "${HERE}/job2_subgraph_analysis.py" --config "${CONFIG}" --task-id "$i"; done
  for i in {0..15}; do python "${HERE}/job3_crystal_baselines.py" --config "${CONFIG}" --task-id "$i"; done
  python "${HERE}/job4_bond_order.py" --config "${CONFIG}" --validate
  for ((i=0;i<J4N;i++)); do python "${HERE}/job4_bond_order.py" --config "${CONFIG}" --task-id "$i"; done
  for ((i=0;i<J5N;i++)); do python "${HERE}/job5_high_force_comparison.py" --config "${CONFIG}" --task-id "$i"; done
  for ((i=0;i<NSIM;i++)); do python "${HERE}/job6_force_cluster_analysis.py" --config "${CONFIG}" --task-id "$i"; done
  python "${HERE}/merge_results.py" --config "${CONFIG}"
}

if [[ "${MODE}" == "--local" ]]; then run_local; exit 0; fi

commands=(
  "sbatch --parsable -A ${ACCOUNT} -p ${PARTITION} -J ruby_j0 -c 1 --mem=8gb -t 01:00:00 -o ${OUT}/logs/job0_%j.log --wrap='source ${CONDA_SH}; conda activate graph_analysis; python ${HERE}/job0_verify_graphs.py --config ${CONFIG}'"
  "sbatch --parsable -A ${ACCOUNT} -p ${PARTITION} -J ruby_j1 -c 2 --mem=12gb -t 02:00:00 -o ${OUT}/logs/job1_%j.log --wrap='source ${CONDA_SH}; conda activate graph_analysis; export MPLCONFIGDIR=/tmp/yfjin/mpl_\$SLURM_JOB_ID; python ${HERE}/job1_global_figures.py --config ${CONFIG}'"
)
if [[ -n "${FEATURE_INPUT}" ]]; then
  commands+=("sbatch --parsable -A ${ACCOUNT} -p ${PARTITION} -J ruby_stats -c 4 --mem=24gb -t 08:00:00 -o ${OUT}/logs/geometry_stats_%j.log --wrap='source ${CONDA_SH}; conda activate graph_analysis; export MPLCONFIGDIR=/tmp/yfjin/mpl_\$SLURM_JOB_ID; python ${HERE}/../compare_periodic_angles.py --input-dir ${FEATURE_INPUT} --output-dir ${FEATURE_OUTPUT} --geometries ${GEOMETRIES}'")
fi

if [[ "${MODE}" == "--dry-run" ]]; then
  printf '%s\n' "${commands[@]}"
  echo "sbatch [common] -J ruby_j2fast -c 1 --mem=4gb -t 01:00:00 --array=0-$((J2BLOCK-1))%24 --dependency=afterok:<JOB0> ... job2_subgraph_analysis.py --workers 1"
  echo "sbatch [common] -J ruby_j2paths -c 4 --mem=8gb -t 03:00:00 --array=${J2BLOCK}-$((2*J2BLOCK-1))%16 --dependency=afterok:<JOB0> ... job2_subgraph_analysis.py --workers 4"
  echo "sbatch [common] -J ruby_j2loops -c 8 --mem=12gb -t 04:00:00 --array=$((2*J2BLOCK))-$((3*J2BLOCK-1))%12 --dependency=afterok:<JOB0> ... job2_subgraph_analysis.py --workers 8"
  echo "sbatch [common] -J ruby_j2spectral -c 8 --mem=12gb -t 06:00:00 --array=$((3*J2BLOCK))-$((4*J2BLOCK-1))%12 --dependency=afterok:<JOB0> ... job2_subgraph_analysis.py --workers 8"
  echo "sbatch [common] -J ruby_j2conn -c 4 --mem=8gb -t 04:00:00 --array=$((4*J2BLOCK))-$((5*J2BLOCK-1))%16 --dependency=afterok:<JOB0> ... job2_subgraph_analysis.py --workers 4"
  echo "sbatch [common] -J ruby_j3 -c ${SETTINGS[7]} --mem=${SETTINGS[8]} -t ${SETTINGS[9]} --array=0-15%8 --dependency=afterok:<JOB0> ... job3_crystal_baselines.py"
  echo "sbatch [common] -J ruby_j4validate -c 2 --mem=8gb -t 01:00:00 --dependency=afterok:<JOB0> ... job4_bond_order.py --validate"
  echo "sbatch [common] -J ruby_j4 -c ${SETTINGS[10]} --mem=${SETTINGS[11]} -t ${SETTINGS[12]} --array=0-$((J4N-1))%${SETTINGS[13]} --dependency=afterok:<JOB0>:<VALIDATE> ... job4_bond_order.py"
  echo "sbatch [common] -J ruby_j5complete -c ${SETTINGS[15]} --mem=${SETTINGS[16]} -t ${SETTINGS[17]} --array=0-$((NSIM-1))%${SETTINGS[18]} --dependency=afterok:<JOB0> ... job5_high_force_comparison.py"
  echo "sbatch [common] -J ruby_j5center -c ${SETTINGS[15]} --mem=${SETTINGS[16]} -t ${SETTINGS[17]} --array=${NSIM}-$((J5N-1))%${SETTINGS[18]} --dependency=afterok:<ALL_JOB2> ... job5_high_force_comparison.py"
  echo "sbatch [common] -J ruby_j6 -c ${SETTINGS[19]} --mem=${SETTINGS[20]} -t ${SETTINGS[21]} --array=0-$((NSIM-1))%${SETTINGS[22]} --dependency=afterok:<JOB0> ... job6_force_cluster_analysis.py"
  echo "sbatch [common] -J ruby_merge -c 4 --mem=32gb -t 04:00:00 --dependency=afterok:<ALL_ARRAYS> ... merge_results.py"
  exit 0
fi

if [[ "${MODE}" == "--submit-new" ]]; then
  J5_COMPLETE=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j5complete -c "${SETTINGS[15]}" --mem="${SETTINGS[16]}" -t "${SETTINGS[17]}" --array="0-$((NSIM-1))%${SETTINGS[18]}" -o "${OUT}/logs/job5_complete_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job5_high_force_comparison.py --config ${CONFIG}")
  J5_CENTER=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j5center -c "${SETTINGS[15]}" --mem="${SETTINGS[16]}" -t "${SETTINGS[17]}" --array="${NSIM}-$((J5N-1))%${SETTINGS[18]}" -o "${OUT}/logs/job5_center_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job5_high_force_comparison.py --config ${CONFIG}")
  J6=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j6 -c "${SETTINGS[19]}" --mem="${SETTINGS[20]}" -t "${SETTINGS[21]}" --array="0-$((NSIM-1))%${SETTINGS[22]}" -o "${OUT}/logs/job6_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job6_force_cluster_analysis.py --config ${CONFIG}")
  MERGE=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_merge56 -c 4 --mem=32gb -t 04:00:00 --dependency="afterok:${J5_COMPLETE}:${J5_CENTER}:${J6}" -o "${OUT}/logs/merge56_%j.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export MPLCONFIGDIR=/tmp/yfjin/mpl_\$SLURM_JOB_ID; python ${HERE}/merge_results.py --config ${CONFIG} --allow-incomplete")
  printf 'Submitted new stages job5=[%s,%s] job6=%s merge=%s\n' "$J5_COMPLETE" "$J5_CENTER" "$J6" "$MERGE"; exit 0
fi

if [[ "${MODE}" != "--submit" ]]; then echo "Usage: $0 --dry-run|--local|--submit-new|--submit" >&2; exit 2; fi
if ! timeout 15s scontrol ping >/dev/null 2>&1; then
  echo "Slurm controller is unavailable; no jobs were submitted." >&2
  exit 1
fi
JOURNAL="${OUT}/logs/submission_$(date +%Y%m%d_%H%M%S).txt"
record_job() { printf '%s=%s\n' "$1" "$2" | tee -a "${JOURNAL}"; }
J0=$(eval "${commands[0]}"); record_job job0 "$J0"
J1=$(eval "${commands[1]} --dependency=afterok:${J0}"); record_job job1 "$J1"
STATS=""; if (( ${#commands[@]} > 2 )); then STATS=$(eval "${commands[2]}"); record_job geometry_stats "$STATS"; fi
J2_FAST=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j2fast -c 1 --mem=4gb -t 01:00:00 --array="0-$((J2BLOCK-1))%24" --dependency="afterok:${J0}" -o "${OUT}/logs/job2_fast_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job2_subgraph_analysis.py --config ${CONFIG} --workers 1")
record_job job2_fast "$J2_FAST"
J2_PATHS=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j2paths -c 4 --mem=8gb -t 03:00:00 --array="$((J2BLOCK))-$((2*J2BLOCK-1))%16" --dependency="afterok:${J0}" -o "${OUT}/logs/job2_paths_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job2_subgraph_analysis.py --config ${CONFIG} --workers 4")
record_job job2_paths "$J2_PATHS"
J2_LOOPS=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j2loops -c 8 --mem=12gb -t 04:00:00 --array="$((2*J2BLOCK))-$((3*J2BLOCK-1))%12" --dependency="afterok:${J0}" -o "${OUT}/logs/job2_loops_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job2_subgraph_analysis.py --config ${CONFIG} --workers 8")
record_job job2_loops "$J2_LOOPS"
J2_SPECTRAL=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j2spectral -c 8 --mem=12gb -t 06:00:00 --array="$((3*J2BLOCK))-$((4*J2BLOCK-1))%12" --dependency="afterok:${J0}" -o "${OUT}/logs/job2_spectral_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job2_subgraph_analysis.py --config ${CONFIG} --workers 8")
record_job job2_spectral "$J2_SPECTRAL"
J2_CONN=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j2conn -c 4 --mem=8gb -t 04:00:00 --array="$((4*J2BLOCK))-$((5*J2BLOCK-1))%16" --dependency="afterok:${J0}" -o "${OUT}/logs/job2_conn_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job2_subgraph_analysis.py --config ${CONFIG} --workers 4")
record_job job2_connectivity "$J2_CONN"
J3=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j3 -c "${SETTINGS[7]}" --mem="${SETTINGS[8]}" -t "${SETTINGS[9]}" --array="0-15%8" --dependency="afterok:${J0}" -o "${OUT}/logs/job3_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; python ${HERE}/job3_crystal_baselines.py --config ${CONFIG}")
record_job job3 "$J3"
VALIDATE=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j4validate -c 2 --mem=8gb -t 01:00:00 --dependency="afterok:${J0}" -o "${OUT}/logs/job4_validate_%j.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; python ${HERE}/job4_bond_order.py --config ${CONFIG} --validate")
record_job job4_validation "$VALIDATE"
J4=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j4 -c "${SETTINGS[10]}" --mem="${SETTINGS[11]}" -t "${SETTINGS[12]}" --array="0-$((J4N-1))%${SETTINGS[13]}" --dependency="afterok:${J0}:${VALIDATE}" -o "${OUT}/logs/job4_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python ${HERE}/job4_bond_order.py --config ${CONFIG}")
record_job job4 "$J4"
J5_COMPLETE=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j5complete -c "${SETTINGS[15]}" --mem="${SETTINGS[16]}" -t "${SETTINGS[17]}" --array="0-$((NSIM-1))%${SETTINGS[18]}" --dependency="afterok:${J0}" -o "${OUT}/logs/job5_complete_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; python ${HERE}/job5_high_force_comparison.py --config ${CONFIG}")
record_job job5_complete "$J5_COMPLETE"
J5_CENTER=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j5center -c "${SETTINGS[15]}" --mem="${SETTINGS[16]}" -t "${SETTINGS[17]}" --array="${NSIM}-$((J5N-1))%${SETTINGS[18]}" --dependency="afterok:${J2_FAST}:${J2_PATHS}:${J2_LOOPS}:${J2_SPECTRAL}:${J2_CONN}" -o "${OUT}/logs/job5_center_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; python ${HERE}/job5_high_force_comparison.py --config ${CONFIG}")
record_job job5_centered "$J5_CENTER"
J6=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_j6 -c "${SETTINGS[19]}" --mem="${SETTINGS[20]}" -t "${SETTINGS[21]}" --array="0-$((NSIM-1))%${SETTINGS[22]}" --dependency="afterok:${J0}" -o "${OUT}/logs/job6_%A_%a.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; python ${HERE}/job6_force_cluster_analysis.py --config ${CONFIG}")
record_job job6 "$J6"
MERGE=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J ruby_merge -c 4 --mem=32gb -t 04:00:00 --dependency="afterok:${J1}:${J2_FAST}:${J2_PATHS}:${J2_LOOPS}:${J2_SPECTRAL}:${J2_CONN}:${J3}:${J4}:${J5_COMPLETE}:${J5_CENTER}:${J6}" -o "${OUT}/logs/merge_%j.log" --wrap="source ${CONDA_SH}; conda activate graph_analysis; export MPLCONFIGDIR=/tmp/yfjin/mpl_\$SLURM_JOB_ID; python ${HERE}/merge_results.py --config ${CONFIG}")
record_job merge "$MERGE"
printf 'Submitted stats=%s job0=%s job1=%s job2=[%s,%s,%s,%s,%s] job3=%s validation=%s job4=%s job5=[%s,%s] job6=%s merge=%s\n' "$STATS" "$J0" "$J1" "$J2_FAST" "$J2_PATHS" "$J2_LOOPS" "$J2_SPECTRAL" "$J2_CONN" "$J3" "$VALIDATE" "$J4" "$J5_COMPLETE" "$J5_CENTER" "$J6" "$MERGE"
