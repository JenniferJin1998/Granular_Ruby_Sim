# Granular ruby simulation analysis

This repository builds particle-contact graphs and analyzes granular ruby simulations. Raw simulation arrays live on permanent storage; generated graphs, tables, figures, and job artifacts live under `AnalysisResults/`.

## Data locations

The canonical raw-data root is:

```text
/nfs/turbo/meche-abucsek/Yuefeng/Granular_Project/Simulation/
├── BoundaryAngle_Container/
│   ├── FinalLoad/Degree{0,15,30,45}/
│   └── Jamming/Degree{0,15,30,45}/
└── BoundaryAngle_Periodic/
    ├── 0Degree/
    └── 30Degree/
```

The graph-generation defaults read `BoundaryAngle_Periodic` and write under the `2026-08-03` results tag. Override the source with `GRAPHPIPE_BASE_PATH`, the common raw root with `GRANULAR_SIMULATION_ROOT`, or the output tag with `GRANULAR_RESULTS_RUN_TAG`.

### Raw-data inventory

| Dataset | Angles | Simulations per angle | Particles per simulation | Force rows by angle |
|---|---|---:|---:|---|
| Container, final load | 0°, 15°, 30°, 45° | 20 | 464 | 61,666; 61,623; 61,936; 61,077 |
| Container, jamming | 0°, 15°, 30°, 45° | 20 | 464 | 42,622; 43,567; 41,552; 41,320 |
| Periodic boundary | 0°, 30° | 20 | 1,575 | 161,542; 158,418 |

Each angle has compatible `forces_collect`, `f_lengths`, `Pos_collect`, and `sigma_collect` arrays. No ROI file is present, so the graph builder will assign the ROI flag as zero unless one is supplied. Final-load folders also contain force-displacement arrays and small fit/reference files.

## Repository guide

| Path | Purpose |
|---|---|
| `AnalysisScripts/Pipeline/` | Restartable graph construction and feature calculations. |
| `AnalysisScripts/RunWhole/` | Monolithic graph-generation workflow and feature reference. |
| `AnalysisScripts/PostAnalysis/` | Comparisons, plotting, boundary-layer, and local-structure analyses. |
| `AnalysisScripts/PostAnalysis/PeriodicRubyPipeline/` | Restartable neighborhood, bond-order, high-force, and cluster analyses. |
| `AnalysisScripts/OldVersions/` | Historical code retained for provenance; not a current entry point. |
| `AnalysisScripts/jobs/` | SLURM submission scripts; generated logs and temporary state are ignored. |
| `AnalysisResults/` | Generated outputs. See its README for the canonical layout and status. |

## Current entry points

Generate periodic-boundary graph features with the staged pipeline:

```bash
cd AnalysisScripts
bash jobs/submit_graph_pipeline_production_all.sh
```

Run boundary-layer analysis for all configured datasets:

```bash
python AnalysisScripts/PostAnalysis/analyze_boundary_layers.py
```

The primary graph view retains actual particle nodes and particle-particle contacts. Wall placeholders and wall-contact edges remain in the full graph where generated, but primary property analysis uses the particle-only core graph unless an output explicitly says `with_walls`.
