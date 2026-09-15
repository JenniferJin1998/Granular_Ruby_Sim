# Periodic Ruby Topology and Bond-Order Pipeline

This restartable pipeline analyzes the saved 0° and 30° graphs without rebuilding or modifying them. The primary view is `core`: all particle nodes and particle-particle contacts. Wall nodes, wall contacts, and properties ending in `_with_walls` are excluded from primary calculations.

## Environment

```bash
source /home/yfjin/Research/anaconda3/etc/profile.d/conda.sh
conda activate graph_analysis
cd /scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608/AnalysisScripts/PostAnalysis/PeriodicRubyPipeline
```

`config.yaml` uses JSON syntax, which is valid YAML, so compute jobs do not require PyYAML. Change paths, workers, thresholds, or SLURM resources there before submission.

The same code supports the four final-load geometries. Its separate configuration uses `0deg`, `15deg`, `30deg`, and `45deg`, with 20 simulations per geometry, and writes restartable calculations into `AnalysisResults/final_load/1_network_property_comparison/artifacts/local_structure_pipeline`. It writes feature distributions and simulation-mean boxplots into `AnalysisResults/final_load/1_network_property_comparison/feature_properties`; all six geometry pairs are tested.

```bash
./submit_final_load_state.sh --dry-run
./submit_final_load_state.sh --submit
```

The final-load source pickle and existing feature CSV files are read only. Job 0 currently identifies no periodic axes for these explicitly wall-bounded samples. Primary Jobs 1–6 use the particle-only `core` graph and exclude `_with_walls` properties; the separate feature-distribution stage intentionally mirrors the earlier feature-table analysis and therefore includes full-table edge rows, including wall contacts tagged by `is_wall_contact`.

Local betweenness uses a deterministic approximation with at most 32 source nodes (configured by `job2_betweenness_approx_k`). Exact edge connectivity is attempted only through the configured 250-node neighborhood limit; larger neighborhoods are retained with `edge_connectivity=NaN` and `connectivity_exact_computed=false`. These choices prevent a few overlapping large neighborhoods from dominating the complete array while making every omission explicit.

## Jobs

- `job0_verify_graphs.py`: read-only graph/property inventory and estimated periodic geometry.
- `job1_global_figures.py`: simulation-replicate topology evidence.
- `job2_subgraph_analysis.py`: restartable `bundle × simulation × hop` tasks (800 for the periodic dataset; 1,600 for final load). Submission uses five resource-matched arrays: fast (1 CPU), paths (4), fundamental cycles (8), sparse spectral (8), and connectivity (4). Local loop fractions use a fundamental cycle basis because repeating minimum-cycle-basis optimization for every overlapping 5-hop neighborhood is computationally prohibitive.
- `job3_crystal_baselines.py`: 16 `crystal × hop` tasks.
- `job4_bond_order.py`: one task per simulation plus 4 ideal-crystal tasks.
- `job5_high_force_comparison.py`: one complete-graph task and four centered-neighborhood tasks per simulation. Stored `is_force_chain_node` and `is_high_force` labels are reused. Centered tasks reuse Job 2 topology rows and aggregate all eligible scalar node/edge properties in the identical induced neighborhoods.
- `job6_force_cluster_analysis.py`: one task per simulation. A force cluster is one connected component of the network containing only high-force particle-particle edges and their endpoints; isolated particles are excluded.
- `plot_basic_high_force_systems.py`: deterministic random 0°/30° system selections shown along x, y, z, and in perspective, with stored high-force nodes/edges in red and all other particles/contacts in gray.
- `merge_results.py`: completion audit, merged tables, replicate-level tests, and Job 5/6 figures.
- `plot_job2_results.py`: simulation-point hop comparisons, six-statistic trends, effect-size/p-value heatmaps, and local-versus-complete plots from the merged Job 2 tables.
- `analyze_bivariate_relationships.py`: corrected-PBC two-dimensional
  probability bins, q-order distributions, and simulation-mean comparisons,
  all with property-wise shared axes, bins, and color limits.

Outputs are written atomically. A task is skipped only when both its output and matching `.complete.json` manifest exist and use the current configuration and pipeline-code hashes.

## Safe validation sequence

```bash
python job0_verify_graphs.py --dry-run
python job0_verify_graphs.py
python job1_global_figures.py --dry-run
python job2_subgraph_analysis.py --task-id 0 --dry-run
python job3_crystal_baselines.py --task-id 0 --dry-run
python job4_bond_order.py --task-id 0 --dry-run
python job4_bond_order.py --validate --dry-run
python job5_high_force_comparison.py --task-id 0 --dry-run
python job5_high_force_comparison.py --task-id 40 --dry-run
python job6_force_cluster_analysis.py --task-id 0 --dry-run
python plot_basic_high_force_systems.py
./submit_jobs.sh --dry-run
```

Pilot one ruby neighborhood, one crystal neighborhood, and bond order:

```bash
python job2_subgraph_analysis.py --task-id 0 --workers 4
python job3_crystal_baselines.py --task-id 0
python job4_bond_order.py --validate
python job4_bond_order.py --task-id 0
python job4_bond_order.py --task-id 40
python merge_results.py --allow-incomplete
```

## Full execution

Local execution (large and not recommended on a login node):

```bash
./submit_jobs.sh --local
```

Print SLURM commands without submitting:

```bash
./submit_jobs.sh --dry-run
```

Submit the dependency chain:

```bash
./submit_jobs.sh --submit
```

When Jobs 0–4 and all Job 2 neighborhood files already exist, submit only the new stages and their merge:

```bash
./submit_jobs.sh --submit-new
```

`--submit-new` does not resubmit earlier arrays. The centered Job 5 worker checks that every required Job 2 bundle file exists before it starts. Its final merge allows an already-known missing older-stage artifact (currently one Job 3 crystal summary) but still waits for every new Job 5/6 array to finish successfully; the completion audit records exact counts.

Do not submit until pilot timing confirms the configured Job 2 bundle resources. Overlapping neighborhoods are summarized within simulations; angle-level inference uses simulations, never particles or neighborhoods, as independent replicates.

## High-force conventions

Final-load and periodic results now retain two explicit force splits:

- `force_split1`: `2 * mean(normal_force)` over all pooled 0deg contacts,
  including particle-particle and wall contacts.
- `force_split2`: `2 * mean(normal_force)` over pooled 0deg
  particle-particle contacts only.

Each reference threshold is fixed from 0deg and applied to every geometry and
to every full-graph contact. Thus, in split 2, wall contacts do not influence
the numerical threshold but can still be labeled high-force. Particle node
labels count incident high-force particle and wall contacts, matching the graph
builder's established semantics. Job 6 connected clusters always use only
high-force particle-particle edges. Jamming intentionally retains the
final-load split-1 threshold; because that threshold selects zero jamming
contacts, no connected-cluster analysis is presented for jamming.

The configured edge label is `is_high_force` and the configured node label is `is_force_chain_node`. The node label is read as stored. Only if it is absent does Job 5 classify a particle as high-force when it is incident to at least one high-force particle-particle edge; each manifest records whether the stored or fallback rule was used.

In the current saved graphs the stored node classification is intentionally not reconstructed from core-edge incidence: it contains 34–58 additional labeled particles per simulation and misses no endpoint of a high-force core edge. Job 5 preserves that stored classification and records both discrepancy counts in each complete-task manifest/table. Job 6 follows its different explicit definition and therefore builds clusters only from endpoints of `is_high_force` particle-particle edges.

Job 5 reports pooled distributions descriptively, but tests high-force/non-high-force contrasts through paired simulation summaries and tests angle effects through the 20 independent simulation contrasts per angle. Centered neighborhoods overlap, so centers are never treated as independent replicates.

Job 6 unwraps every connected cluster by traversing its contacts with minimum-image displacements using the corrected configuration-exact x/y-periodic geometry. Shape metrics therefore do not split seam-crossing clusters. With x and y periodic, boundary/loading-surface distances refer to the nonperiodic z surfaces, and `loading_axis: 2` records that convention. Cluster pooling is used only for descriptive plots; 0°/30° tests use simulation summaries.

## Bond-order conventions

`scipy.special.sph_harm(m, l, phi, theta)` is used with SciPy's complex Condon–Shortley convention: `theta` is the azimuth and `phi` is the polar angle in that API's historical naming. The code passes the computed azimuth as SciPy's `theta` argument and the polar angle as `phi`. Each undirected particle contact contributes two directed bonds, one to each endpoint's local environment. Global coherent order averages those directed endpoint contributions once; it does not add another undirected-edge weighting. Particles without particle-particle neighbors receive `NaN` for local and neighbor-averaged invariants.

Ruby minimum-image vectors use the orthogonal x/y-periodic box reconstructed by Job 0. Ideal HCP uses a rectangular periodic representation of the correct three-dimensional hexagonal lattice with ideal layer spacing and six A/B layers.

Ideal-crystal local metrics are evaluated once per symmetry-equivalent `(z plane, coordination)` environment and expanded back to one row per particle. This avoids recalculating identical lateral-periodic neighborhoods hundreds of times while preserving particle-weighted distributions.

Every overlaid or multi-panel distribution comparison uses one pooled set of
bin edges per property. Consequently, bin widths and boundaries are identical
for all angles, force groups, boundary groups, and crystal references shown in
the same comparison; bins are never selected independently per curve.

## Exact periodic-geometry correction

The raw periodic simulations use exact box lengths `Lx=0.0018 m` and
`Ly=0.0030 m`, with periodicity in x and y only. The earlier Job 0 estimate
added a contact diameter to a seam separation and therefore overestimated both
lengths. The corrected graph in `0_graph_and_basic_stats/graph_data/` is now
canonical; the prior results are preserved in `_archive_pre_periodic_correction/`.

```bash
cd /scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim_202608
python AnalysisScripts/PostAnalysis/correct_periodic_geometry.py

cd AnalysisScripts/PostAnalysis/PeriodicRubyPipeline
python job0_verify_graphs.py --config config_periodic_corrected.yaml
python job4_bond_order.py --config config_periodic_corrected.yaml --validate
python job4_bond_order.py --config config_periodic_corrected.yaml --all
python job6_force_cluster_analysis.py --config config_periodic_corrected.yaml --all
python merge_results.py --config config_periodic_corrected.yaml --allow-incomplete
python analyze_bivariate_relationships.py --config config_periodic_corrected.yaml
```

The corrected graph stores the original contact angle as
`angle_with_zz_original`. Its canonical `angle_with_zz` and audit field
`angle_with_zz_periodic` are calculated from the exact minimum-image center
displacement. Per-contact periodic displacement, shift, distance, and seam
flags are also stored. Wall labels `-1/-2/-3` identify the bottom surface and
`-4/-5` identify the top surface.
