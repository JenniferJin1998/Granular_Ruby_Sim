# Core-only force-cluster analysis (final load, force_split2)

Writes `AnalysisResults/final_load/04_force_cluster_analysis_core_only/`; each
numbered script fills the results folder with the same number. Labels come from the
collaborator workbook; see that folder's README for definitions.

Step 00 needs openpyxl, which the `graph_analysis` environment lacks, so it runs
with the base Anaconda Python. The other steps read only the CSVs it exports.

```bash
cd AnalysisScripts/PostAnalysis/ForceClusterCoreOnly
/home/yfjin/Research/anaconda3/bin/python 00_export_workbook_labels.py
P=/home/yfjin/Research/anaconda3/envs/graph_analysis/bin/python
$P 01_betweenness_vs_hydrostatic_stress.py
$P 02_global_betweenness_rank.py      # --bin-percent 10 for coarser bins
$P 03_local_betweenness_rank.py
$P 04_spatial_comparison.py         # --layers, --top-percent
$P 05_correlation.py                # about 6 min (edge betweenness)
# 06 needs scikit-learn: gnn_granular env; about 11 min with 8 CPUs
sbatch ../../jobs/run_core_only_feature_embedding.slurm
```

Step 00 stops if the workbook labels, betweenness, or contacts disagree with
`0_graph_and_basic_stats/force_splits/force_split2/node_features.csv`. Shared paths,
colors, and the per-system 95% interval live in `core_only_common.py`.
