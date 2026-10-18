# Post-analysis layout

- `ForceCluster_vf0.ipynb` and `PresentationFigurePlots.ipynb` are the original notebooks.
- `organized_notebooks/01_analysis_draft_workflow.ipynb` is the clean copy for analysis, force-cluster extraction, draft plots, and exploratory calculations.
- `organized_notebooks/02_publication_figure_workflow.ipynb` is the clean copy for polished final plots.
- `organized_notebooks/03_final_vs_jamming_comparison.ipynb` compares final-load and jamming-state graph properties.
- `organized_notebooks/04_jamming_final_threshold_node_groups.ipynb` analyzes corrected jamming-state node properties grouped by final-state high-force node labels.
- `relabel_jamming_with_final_threshold.py` creates the corrected jamming graph list using the final-load `0deg` high-force threshold.
- `compare_load_states.py` writes node-level and mean-change comparison tables.
- `transfer_final_force_node_groups.py` transfers final-state high-force node labels onto jamming node features and writes grouped node-property stats.
- `force_threshold_percolation.py` performs the Liu et al. (2023)-inspired
  per-simulation strong-contact threshold sweep for final-load and periodic
  graphs. It stores results under each dataset's
  `3_force_threshold_percolation/` folder and combined comparisons under
  `AnalysisResults/cross_state/3_force_threshold_percolation/`.
- `force_threshold_percolation_config.json` records the sweep (`n=0.0` through
  `5.0` by `0.1`), graph inputs, wall labels, and exact periodic lengths.

Submit the 80 worker tasks and dependent summary job with:

```bash
bash AnalysisScripts/jobs/submit_force_threshold_percolation.sh
```

The strong network contains particle-particle contacts only, and its threshold
uses the mean particle-particle normal force of the same simulation. The main
selected `n` is the last sampled value for which the largest cluster spans from
the bottom-wall contact set to the top-wall contact set.

## Final-load force-split-2 particle-only workbook

Run `python AnalysisScripts/PostAnalysis/export_final_load_force_split2_tables.py`
with pandas and XlsxWriter installed to export particle-only
populations for all four geometries and all 20 simulations per geometry. The
workbook is saved under
`AnalysisResults/final_load/2_force_cluster_comparison/force_split2/final_load_force_split2_populations.xlsx`.
Its `nodes` sheet contains particle-only force-chain labels, degree, closeness, and
betweenness. Its `edges` sheet contains high-force labels, normal force, contact
angle to z in degrees, and both endpoint nodes' labels and centralities. Only
particle nodes and particle-particle contacts are included. Node force labels
and high-force degrees are recomputed using only retained contacts, excluding
wall-contact effects present in the existing stored Job 5 node classification.
Properties are copied from the complete core graph, rather than recalculated
on the high-force population. A companion JSON records sources, counts,
threshold verification, and the distribution plotting conventions.

## Force-threshold property distributions

`force_threshold_property_distributions.py` sweeps `n=0:0.025:4` for final-load
data. The force cutoff is `n` times the pooled 0-degree particle-particle mean
normal force, fixed across all geometries and simulations. Wall nodes and wall
contacts are excluded throughout. A particle is high-force if it is incident
to at least one selected particle-particle contact. Stored wall-influenced
node labels are not used.

Run with the existing plotting environment:

```bash
/home/yfjin/Research/anaconda3/envs/graph_analysis/bin/python \
  AnalysisScripts/PostAnalysis/force_threshold_property_distributions.py
```

Results are saved in
`AnalysisResults/final_load/2_force_cluster_comparison/force_threshold_property_distributions/`:

- `tables/pooled_geometry_group_summaries.csv`: main plotted means and medians,
  with group counts and fractions for every geometry, threshold, and property.
- `tables/per_simulation_group_summaries.csv`: the same calculations separately
  for each simulation, preserving the simulation as the independent replicate.
- `tables/simulation_balanced_summaries.csv`: equal-weight averages of available
  simulation means/medians, with the number of nonempty simulation groups.
- `tables/particle_input_properties.csv`, `particle_contact_input_properties.csv`,
  and `thresholds.csv`: particle-only input values and numerical force cutoffs.
- `figures/threshold_curves/`: one four-property figure per geometry and statistic.
- `figures/geometry_lines/`: high/non-high figures with all thresholds as lines;
  a shared colorbar identifies `n`, and integer thresholds receive heavier lines.
- `figures/geometry_threshold_surfaces/`: paired high/non-high surfaces over
  geometry and threshold, one four-property figure for each statistic.
- `analysis_manifest.json`: input and script hashes, reference mean, definitions,
  validation, and figure paths.

All three plot types are generated for means and medians in PDF, PNG, and SVG.
The plots pool all particle/contact observations within each geometry and force
group, matching the descriptive distribution convention. Degree, closeness, and
betweenness are copied from the complete particle-only graph and held fixed;
only population membership changes. Contact angle to z is an edge property in
degrees. Empty populations have undefined means/medians and appear as gaps.
The script validates exhaustive partitions, monotone group sizes, exact particle
degree against contact incidence, and agreement of pooled summaries with
count-weighted simulation means. These summaries measure distribution location;
they do not characterize the entire distribution or provide significance tests.

Options include `--start`, `--stop`, `--step`, `--input-dir`, `--output-root`,
`--formats`, `--dpi`, and `--tables-only`. Use a separate `--output-root` when
comparing different sweep configurations. Rerunning overwrites matching outputs.

Selected-threshold comparisons plot geometry angle versus pooled property mean
with high-force and non-high-force curves together. Defaults are
`n=0.5,1,1.5,1.75,2,2.25,2.5,2.75,3,3.5`, with one four-property figure per
threshold and shared property-wise y limits across these figures. They are
included in a full run. To add or refresh only these figures from saved tables:

```bash
/home/yfjin/Research/anaconda3/envs/graph_analysis/bin/python \
  AnalysisScripts/PostAnalysis/force_threshold_property_distributions.py --selected-only
```

Figures are saved in `figures/geometry_high_vs_non_high/mean/`, plotted values
in `tables/selected_threshold_geometry_means.csv`, and provenance in
`selected_threshold_geometry_mean_manifest.json`. Override the selections with
`--selected-thresholds` followed by multipliers present in the saved sweep.

## Crystal-reference networks

- `crystal_reference_networks.py` converts the existing deterministic
  SC/BCC/FCC/HCP bond-order references into full contact graphs, adds periodic
  x/y geometry and top/bottom boundary labels, calculates the same graph
  feature families as the periodic Ruby pipeline, and produces crystal-only
  and Ruby-versus-crystal property comparisons.
- `crystal_reference_config.json` records the particle diameter, contact rule,
  structures, boundary labels, input Ruby tables, and output root.
- `plot_crystal_reference_geometry.py` renders each full crystal as x/y/z
  projections plus a 3D perspective, and documents its primitive unit cell,
  lattice vectors, basis, particle-center equation, and familiar conventional
  cubic/hexagonal cell. Both primitive and conventional cells also receive
  four-view figures (projections along x/y/z plus a 3D perspective). Solid blue
  lines are physical particle contacts, magenta lines in full-system views are
  contacts wrapped through periodic x/y boundaries, and dashed gray lines are
  cell or simulation-box guides rather than graph edges.
- `plot_crystal_reference_properties.py` adds four-panel shared-bin
  distributions, pooled node/contact boxplots, whole-graph bar charts, and
  annotated standardized property-mean heatmaps to the finalizer's overlay
  plots. It also produces annotated node/edge count, mean, and standard-
  deviation heatmaps at graph distances 0, 1, and 2 from combined top+bottom,
  top-only, and bottom-only boundaries.

Submit the restartable overnight graph-property arrays and dependent finalizer
with:

```bash
bash AnalysisScripts/jobs/submit_crystal_reference_networks.sh
```

If only the long-running BCC/FCC/HCP minimum-cycle-basis patches are missing,
resume those three jobs and chain the finalizer with:

```bash
bash AnalysisScripts/jobs/submit_crystal_reference_resume.sh
```

The results use `AnalysisResults/crystal_references/` with numbered sections
0, 1, and 3. Section 2 is omitted because the ideal structures have no force
field and therefore no high-force or force-cluster population.
