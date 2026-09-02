#!/bin/bash
set -eo pipefail

SCRIPT_DIR="/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisScripts"
cd "${SCRIPT_DIR}"

mkdir -p jobs/logs

BUILD_JOB=$(sbatch --parsable jobs/submit_graph_pipeline_production_build.sh)
echo "Submitted build job: ${BUILD_JOB}"

TOPO_JOB=$(GRAPHPIPE_PROPERTY=topology sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name=gp_topo_prod \
    --array=0-39%20 \
    --cpus-per-task=1 \
    --mem=8gb \
    --time=02:00:00 \
    jobs/submit_graph_pipeline_production_property_array.sh)
echo "Submitted topology array: ${TOPO_JOB}"

LOOP_JOB=$(GRAPHPIPE_PROPERTY=loop sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name=gp_loop_prod \
    --array=0-39%20 \
    --cpus-per-task=1 \
    --mem=16gb \
    --time=20:00:00 \
    jobs/submit_graph_pipeline_production_property_array.sh)
echo "Submitted loop array: ${LOOP_JOB}"

PAIR_JOB=$(GRAPHPIPE_PROPERTY=pair_edge sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name=gp_pair_prod \
    --array=0-39%6 \
    --cpus-per-task=16 \
    --mem=32gb \
    --time=08:00:00 \
    jobs/submit_graph_pipeline_production_property_array.sh)
echo "Submitted pair-edge array: ${PAIR_JOB}"

NODE_JOB=$(GRAPHPIPE_PROPERTY=node_connectivity sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name=gp_node_prod \
    --array=0-39%12 \
    --cpus-per-task=8 \
    --mem=32gb \
    --time=03:00:00 \
    jobs/submit_graph_pipeline_production_property_array.sh)
echo "Submitted node-connectivity array: ${NODE_JOB}"

CURV_JOB=$(GRAPHPIPE_PROPERTY=curvature sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name=gp_curv_prod \
    --array=0-39%20 \
    --cpus-per-task=1 \
    --mem=8gb \
    --time=01:00:00 \
    jobs/submit_graph_pipeline_production_property_array.sh)
echo "Submitted curvature array: ${CURV_JOB}"

NFD_JOB=$(GRAPHPIPE_PROPERTY=nfd sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name=gp_nfd_prod \
    --array=0-39%20 \
    --cpus-per-task=1 \
    --mem=8gb \
    --time=01:00:00 \
    jobs/submit_graph_pipeline_production_property_array.sh)
echo "Submitted nfd array: ${NFD_JOB}"

FINAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${TOPO_JOB}:${LOOP_JOB}:${PAIR_JOB}:${NODE_JOB}:${CURV_JOB}:${NFD_JOB} \
    jobs/submit_graph_pipeline_production_finalize.sh)
echo "Submitted finalize job: ${FINAL_JOB}"

echo "Final output directory:"
echo "/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/AnalysisResults/PeriodicBoudaries/GraphPipeline_0deg_30deg"
