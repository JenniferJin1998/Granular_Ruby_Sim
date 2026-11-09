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
$P 07_radial_control.py             # about 1 min; --bin-percent (default 10)
$P 08_radial_band_visuals.py         # about 2 min; --clip 5 95 (color range percentiles), --systems (example sim per angle)
$P 09_radial_auc.py                 # seconds; --bins (default 12)
```

Step 00 stops if the workbook labels, betweenness, or contacts disagree with
`0_graph_and_basic_stats/force_splits/force_split2/node_features.csv`. Shared paths,
colors, and the per-system 95% interval live in `core_only_common.py`.

## Other datasets (`BC_DATASET`)

Steps 01, 02, 05, 07 and 09 also run on other datasets, chosen with the environment variable
`BC_DATASET` (default `final_load_force`, the run above). Definitions and output folders are in
`core_only_common.py`; labels are written first by `00_prepare_dataset_labels.py`.

| `BC_DATASET` | Labels | Output |
|---|---|---|
| `final_load_stress` | high-stress = hydrostatic stress <= system mean - 1 SD | `final_load/04_force_cluster_analysis_core_only/10_high_stress_classification/` |
| `periodic_force` | force cluster (particle-particle contact >= stored force_split2 threshold) | `periodic_boundaries/2026-08-03/5_betweenness_force_cluster_core_only/` (01, 02, 05 only; no radius) |
| `experiment_stress` | high-stress = hydrostatic stress <= sample mean - 1 SD (Results/14 graphs, LS2) | `Granular_RubyExp_202608/Results/18_BetweennessHighStress/` |

```bash
export BC_DATASET=experiment_stress           # or final_load_stress / periodic_force
$P 00_prepare_dataset_labels.py
$P 01_betweenness_vs_hydrostatic_stress.py
$P 02_global_betweenness_rank.py               # experiment: --bin-percent 10 (one sample per angle)
$P 05_correlation.py
$P 07_radial_control.py                        # not for periodic_force
$P 09_radial_auc.py; $P 09_radial_auc.py --bins 6
# periodic_force and final_load_stress together on Slurm (05 is slow for the periodic graphs):
sbatch ../../jobs/run_betweenness_datasets.slurm
```

For the stress datasets the groups are still called `high_force` inside the code; every saved
table and figure is relabeled `high_stress` / "High-stress" (`core_only_common._install_relabeling`).
Step 05 differs by dataset: stress datasets correlate betweenness with compressive stress
(-stress_hydro) and have no contact measure; location control is r and z_eff (final load),
r and height (experiment), height only (periodic).
