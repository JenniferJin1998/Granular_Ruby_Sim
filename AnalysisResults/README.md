# Results catalog

All names use lower-case `snake_case`. The first level identifies the physical dataset; the second identifies an analysis product. Generated content is ignored by Git, while this catalog is tracked.

## Canonical layout

| Path | Role | Current status |
|---|---|---|
| `final_load/graph_features/` | Canonical graph pickle and flat node, edge, and graph feature tables for 0°, 15°, 30°, and 45°. | Populated. |
| `final_load/angle_comparison/` | Derived cross-angle distributions and statistical comparisons. | Populated. |
| `final_load/boundary_layers/` | Derived graph-distance-to-wall maps, tables, and tests. | Populated. |
| `final_load/local_structure/` | Neighborhood, bond-order, high-force, and force-cluster pipeline. | Partially populated: metadata/global figures exist; later jobs are empty. |
| `jamming/graph_features/` | Canonical jamming graph outputs. | Empty. |
| `jamming/graph_features_final_threshold/` | Jamming graphs relabeled with the final-load high-force threshold. | Empty. |
| `jamming/final_force_node_groups/` | Jamming node analysis grouped using final-load force labels. | Empty. |
| `periodic_boundaries/2026-08-03/graph_features/` | Canonical periodic graph pickle, flat feature tables, and restartable graph-build intermediates. | Populated. |
| `periodic_boundaries/2026-08-03/angle_comparison/` | Derived 0° versus 30° distributions and statistical comparisons. | Populated. |
| `periodic_boundaries/2026-08-03/boundary_layers/` | Derived graph-distance-to-boundary maps, tables, and tests. | Populated. |
| `periodic_boundaries/2026-08-03/local_structure/` | Neighborhood, crystal/bond-order, high-force, and force-cluster analysis. | Populated; `combined_results/` contains presentation-ready summaries. |
| `periodic_boundaries/2026-08-03/tests/` | Historical test-run locations. | Currently empty. |
| `cross_state/final_vs_jamming_final_threshold/` | Final-load versus jamming comparisons using the final-load threshold. | Figures only; source jamming tables are currently absent. |

## What is primary

Start with `graph_features/` when another script needs machine-readable input. Start with `angle_comparison/`, `boundary_layers/`, or `local_structure/combined_results/` when reviewing results. Directories named `job0_...` through `job6_...` are restartable pipeline artifacts, and `logs/` is operational history.

## Apparently repeated outputs

Most repeated property names are separate analysis levels rather than duplicates:

- `individual_simulation_distributions/` shows each simulation; `pooled_distributions/` pools particles or contacts; `simulation_level_statistics/` treats simulations as replicates.
- Boundary outputs ending in `_0_vs_rest` and `_0_1_rest` answer different shell-grouping questions.
- Names ending in `_with_walls` use the full graph; the corresponding unsuffixed name uses the particle-only core graph.
- `graph_features/` is the source dataset, while `angle_comparison/`, `boundary_layers/`, and `local_structure/` are different downstream analyses.

One byte-for-byte duplicate set is isolated in `periodic_boundaries/2026-08-03/local_structure/job5_high_force_comparison/pilot_stale_filenames/`. Its three files have canonical copies beside that folder and can be removed if storage cleanup is desired.

The JSON completion manifests and `config.snapshot.json` files record the absolute paths used by their original runs. Those historical paths are provenance, not the current directory names; new runs use the tracked configs under `AnalysisScripts/PostAnalysis/PeriodicRubyPipeline/`.

## Naming policy

Use `<dataset>/<run_date>/<analysis_name>/` for dated runs and avoid implementation names such as `ReusablePipeline`. Put thresholds, geometry choices, code hashes, and other method details in metadata files instead of directory names.
