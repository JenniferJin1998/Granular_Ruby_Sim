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
  cubic/hexagonal cell.
- `plot_crystal_reference_properties.py` adds four-panel shared-bin
  distributions, pooled node/contact boxplots, whole-graph bar charts, and
  standardized property-mean heatmaps to the finalizer's overlay plots.

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
