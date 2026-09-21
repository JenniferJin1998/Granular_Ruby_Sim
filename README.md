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

## Current analysis conventions

The periodic graph uses the exact minimum-image geometry with x/y periodicity,
`Lx = 0.0018 m`, `Ly = 0.0030 m`, and no z wrapping. Contact angles, bond-order
vectors, and force-cluster shape/orientation calculations all use these
periodic displacements. Historical results made with inferred box lengths are
retained only as archived provenance.

Periodic and final-load force-dependent results have two versions:

| Version | Fixed threshold definition |
|---|---|
| `force_split1` | Twice the pooled 0° mean normal force over particle-particle and wall contacts. |
| `force_split2` | Twice the pooled 0° mean normal force over particle-particle contacts only. |

The split-2 thresholds are `1.897454511106544e-05` for periodic simulations
and `0.3764788410084238` for final load. Each threshold is fixed from 0° and
then applied to every geometry and the full contact list. Connected force
clusters always contain particle-particle edges only. Jamming intentionally
uses the final-load split-1 threshold (`0.4200944651809045`); no jamming
contact exceeds it, so connected force-cluster plots are omitted.

Boundary-layer analyses use graph distance from directly wall-contacting
particles. Periodic results provide combined top/bottom, top-only, and
bottom-only references. Final-load and jamming results additionally retain the
original top/bottom/side-wall reference. Each reference includes signed
geometry-minus-0 comparisons (`15°-0°`, `30°-0°`, and `45°-0°` where those
geometries exist).

The force-threshold percolation workflow for final load and periodic data uses
the per-simulation criterion
`normal_force >= n * mean_particle_particle_normal_force` and sweeps
`n=0.0, 0.1, ..., 5.0`. Its primary critical `n` is the last sampled threshold
whose largest strong cluster still spans between the physical bottom- and
top-wall contact sets. Periodic x/y contacts use exact minimum-image geometry;
spanning is assessed along nonperiodic z. The workflow also retains alternate
paper-motivated indicators such as the diameter peak, cluster-count peak,
degree near two, and exact widest-path threshold.

Ideal SC, BCC, FCC, and HCP references can also be treated as complete contact
networks rather than only bond-order benchmarks. These graphs use the Ruby
particle diameter, approximately 1,500–1,600 particles, periodic x/y contacts,
and top/bottom z-surface labels. Their numbered result layout contains graph
construction, whole-network property analysis, and comparison against periodic
Ruby 0°/30°; force-cluster analysis is omitted because ideal lattices do not
define contact forces or particle stresses.

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

Create a particle-particle-reference force split without modifying the source
graph:

```bash
python AnalysisScripts/PostAnalysis/relabel_particle_contact_threshold.py \
  --input-dir <canonical_graph_directory> \
  --output-dir <force_split2_graph_directory>
```

Submit the final-load and periodic force-threshold sweeps plus their dependent
summary job:

```bash
bash AnalysisScripts/jobs/submit_force_threshold_percolation.sh
```

Submit the overnight crystal-reference graph analysis:

```bash
bash AnalysisScripts/jobs/submit_crystal_reference_networks.sh
```

The primary graph view retains actual particle nodes and particle-particle contacts. Wall placeholders and wall-contact edges remain in the full graph where generated, but primary property analysis uses the particle-only core graph unless an output explicitly says `with_walls`.
