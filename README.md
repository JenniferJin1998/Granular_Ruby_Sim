# Granular Ruby simulation analysis

This repository builds and analyzes particle-contact graphs for ruby granular simulations at jamming, final load, and periodic-boundary conditions. The analysis convention is to retain actual particle nodes and particle-particle contacts. Wall placeholders and wall-contact edges remain in the full graph data where generated, but are excluded from primary property analysis.

## Folder guide

| Path | Purpose |
|---|---|
| `Data/` | Simulation inputs grouped into `JammingState`, `FinalLoadState`, and `PeriodicBoudaries`. |
| `AnalysisScripts/Pipeline/` | Core graph construction and feature calculations. |
| `AnalysisScripts/RunWhole/` | Whole-workflow drivers and graph-generation documentation. |
| `AnalysisScripts/PostAnalysis/` | Comparisons, plotting, and reusable downstream analyses. |
| `AnalysisScripts/PostAnalysis/PeriodicRubyPipeline/` | Restartable periodic/final-state analysis jobs. |
| `AnalysisScripts/OldVersions/` | Historical scripts retained for provenance, not the current entry points. |
| `AnalysisScripts/jobs/` | Scheduler scripts, logs, and temporary job state. |
| `AnalysisResults/JammingState/` | Jamming-state graph outputs. |
| `AnalysisResults/FinalLoadState/` | Final-load graphs, geometry comparison, reusable-pipeline results, and the new property/boundary analysis. |
| `AnalysisResults/PeriodicBoudaries/` | Periodic-boundary graphs and angle comparisons, including the new property/boundary analysis. |
| `AnalysisResults/LoadStateComparison/` | Cross-state comparison products. |

`Boudaries` is retained in existing path names for compatibility.

## New property and boundary-layer analysis

The reproducible entry point is [`AnalysisScripts/PostAnalysis/analyze_boundary_layers.py`](AnalysisScripts/PostAnalysis/analyze_boundary_layers.py). It produces:

- enlarged force-cluster node-property maps: force nodes colored by property, force contacts light red, and the remaining network light gray;
- enlarged force-cluster edge-property maps: force contacts colored by property, force nodes light red, and the remaining network light gray;
- perspective and x/y/z projection views for every 3-D property map;
- one common P5/P95 color range across geometry panels for each property;
- graph-distance groups 0, 1, and rest (>=2 or unreachable);
- per-property distributions for `0 vs rest`, followed by separate `0`, `1`, and `rest` curves, with descriptive statistics and contrast tests;
- per-simulation mean and median distributions for groups 0, 1, and rest, compared across boundary angles;
- force-cluster versus non-force histograms and simulation-level mean/median comparisons under both boundary grouping schemes;
- compressed node/edge feature tables with exact boundary distance;
- an audit of boundary assignment for every simulation.

Wall data is used only to seed the boundary definition: a real particle incident to a wall-contact edge is node distance 0. Shortest paths are then calculated only through particle-particle contacts. An analyzed particle contact gets the minimum distance of its two endpoints, so a contact incident to a boundary particle is edge distance 0.

Generated results and detailed inventories are in:

- [`AnalysisResults/PeriodicBoudaries/2026-08-03/GraphPipeline/PropertyBoundaryAnalysis/README.md`](AnalysisResults/PeriodicBoudaries/2026-08-03/GraphPipeline/PropertyBoundaryAnalysis/README.md)
- [`AnalysisResults/FinalLoadState/FullGraph_2mean_ref_geom/PropertyBoundaryAnalysis/README.md`](AnalysisResults/FinalLoadState/FullGraph_2mean_ref_geom/PropertyBoundaryAnalysis/README.md)

Run both datasets from this repository root:

```bash
python AnalysisScripts/PostAnalysis/analyze_boundary_layers.py
```

The default color clipping is P5/P95. For P10/P90 instead:

```bash
python AnalysisScripts/PostAnalysis/analyze_boundary_layers.py \
  --lower-percentile 10 --upper-percentile 90
```

## Environment

The new analysis uses Python 3 with NumPy, pandas, SciPy, and Matplotlib. The broader graph-generation pipeline additionally uses NetworkX and GraphRicciCurvature; see `AnalysisScripts/RunWhole/GraphGeneration_README.md` and `AnalysisScripts/PostAnalysis/PeriodicRubyPipeline/README.md` for pipeline-specific details.
