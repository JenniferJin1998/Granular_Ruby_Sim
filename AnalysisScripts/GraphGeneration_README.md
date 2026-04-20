# GraphGeneration.py Data Structure and Feature Reference

This document describes the outputs written by [GraphGeneration.py](/Users/yfjin/Desktop/PhD/CHESS02142024/GraphNetwork/AnalysisScripts/Graph/GraphGeneration.py), how the graph data are organized in memory, and how each derived property is computed.

It covers the simulation pipeline for the four geometries:

- `0deg`
- `15deg`
- `30deg`
- `45deg`

## 1. Two graph views per simulation

For each simulation slice, the script builds two related undirected graphs:

- `full` graph
  - Contains particle-particle contacts and wall-contact nodes/edges.
  - Particle nodes use integer ids: `0 .. FIXED_NUM_PARTICLES-1`.
  - Wall nodes use string ids such as `0deg_wall_12_5`.
- `core` graph
  - Contains only particle nodes and particle-particle edges.
  - It is the induced subgraph of `full` after removing all wall nodes.

Most derived features follow this naming rule:

- `property`
  - Computed on the core graph without wall nodes.
- `property_with_walls`
  - Computed on the full graph with wall contacts included.

## 2. Main in-memory object

The main Python object saved to disk is:

- `graph_dict_labeled.pkl`

Its structure is:

```python
graph_dict = {
    "0deg": {
        "core": [G_core_sim0, G_core_sim1, ...],
        "full": [G_full_sim0, G_full_sim1, ...],
    },
    "15deg": {...},
    "30deg": {...},
    "45deg": {...},
}
```

Notes:

- `graph_dict[label]["core"][sim_idx]` and `graph_dict[label]["full"][sim_idx]` refer to the same simulation slice in two graph views.
- Ordering is stable and matches the original `contact_counts` order.
- Simulation-level parallelism does not change graph ordering because `joblib.Parallel` returns results in input order.

## 3. Output files written by the script

The script writes the following outputs under `out_path`:

### Core graph objects and bookkeeping

- `graph_dict_labeled.pkl`
  - Pickled `graph_dict` with node, edge, and graph attributes attached.
- `simulation_slices.txt`
  - Text table mapping each `sim_idx` to the original contact/particle slice offsets.
- `high_force_threshold_info.pkl`
  - Threshold mode and threshold values used for high-force labeling.

### High-force export

- `high_force_edges.csv`
  - Edge list of high-force edges after thresholding.

### Pairwise connectivity export

- `pair_edge_connectivity_index.csv`
  - Index of all per-simulation all-pairs connectivity files.
- `pair_edge_connectivity/<geometry>_sim_<id>_pair_edge_connectivity.csv`
  - One file per simulation.
  - Contains all particle-particle pairs, not only existing edges.

### Flat feature tables

- `node_features.csv`
  - One row per core particle node.
- `edge_features.csv`
  - One row per edge in the full graph.
- `graph_features.csv`
  - One row per simulation with scalar graph attributes.
- `graph_feature_arrays.csv`
  - One row per simulation with array-like graph attributes serialized as JSON strings.
- `graph_feature_arrays.pkl`
  - Same content as `graph_feature_arrays.csv`, but with native Python list/dict objects preserved.

### Plot

- `normal_force_distributions_<mode>_<layout>.png`
  - Force histogram panel with the threshold overlaid.

## 4. Node ids and node types

### Particle nodes

- Type: `int`
- Range: `0 .. FIXED_NUM_PARTICLES-1`
- Present in both `core` and `full`

### Wall nodes

- Type: `str`
- Example: `30deg_wall_17_8`
- Present only in `full`
- Each wall contact placeholder becomes a unique wall node in the simulation pipeline used here

## 5. Node attributes

Node attributes fall into two groups: loaded or directly derived from the raw simulation data, and graph-derived metrics.

### 5.1 Loaded / directly derived from raw simulation data

These are assigned during graph construction.

| Attribute | Type | Present on | Description |
|---|---|---|---|
| `position` | `tuple[float, float, float]` | particle nodes | Particle center coordinates |
| `is_wall` | `bool` | all nodes | `False` for particles, `True` for wall nodes |
| `in_center_region` | `bool` | particle nodes | ROI flag from `ROI.mat` |
| `stress_11` | `float` | particle nodes | Normal stress component |
| `stress_22` | `float` | particle nodes | Normal stress component |
| `stress_33` | `float` | particle nodes | Normal stress component |
| `stress_23` | `float` | particle nodes | Shear stress component |
| `stress_13` | `float` | particle nodes | Shear stress component |
| `stress_12` | `float` | particle nodes | Shear stress component |
| `stress_vm` | `float` | particle nodes | von Mises stress computed from the six stress components |
| `stress_hydro` | `float` | particle nodes | Hydrostatic stress = `(s11 + s22 + s33) / 3` |
| `wall_label` | `int` | wall nodes | Original negative wall placeholder id from the contact row |

### 5.2 Graph-derived node metrics

These are computed on `core`, then copied back to the matching particle nodes in `full`.

| Attribute | Type | Meaning |
|---|---|---|
| `degree` | `int` | Degree in core graph |
| `degree_with_walls` | `int` | Degree in full graph |
| `closeness` | `float` | Unweighted closeness centrality in core graph |
| `closeness_with_walls` | `float` | Unweighted closeness centrality in full graph |
| `betweenness` | `float` | Unweighted betweenness centrality in core graph |
| `betweenness_with_walls` | `float` | Unweighted betweenness centrality in full graph |
| `clustering` | `float` | Clustering coefficient in core graph |
| `clustering_with_walls` | `float` | Clustering coefficient in full graph |
| `avg_neighbor_degree` | `float` | Average neighbor degree in core graph |
| `avg_neighbor_degree_with_walls` | `float` | Average neighbor degree in full graph |
| `avg_curvature_no_walls` | `float` | Mean incident Ollivier-Ricci edge curvature from core graph |
| `avg_curvature_with_walls` | `float` | Mean incident Ollivier-Ricci edge curvature from full graph, restricted to particle-particle core edges |
| `principal_eigenvector` | `float` | Component of the leading adjacency eigenvector in core graph |
| `principal_eigenvector_with_walls` | `float` | Component of the leading adjacency eigenvector in full graph |
| `fiedler` | `float` | Component of the Laplacian Fiedler vector in core graph |
| `fiedler_with_walls` | `float` | Component of the Laplacian Fiedler vector in full graph |
| `nfd` | `float` | Node-level NFD slope from the shortest-path shell growth curve |
| `nfd_r2` | `float` | Coefficient of determination for the NFD log-log fit |
| `high_force_degree` | `int` | Number of incident high-force edges, counted on full graph |
| `is_force_chain_node` | `bool` | `True` if `high_force_degree >= 1` |
| `force_chain_role` | `str` | `none`, `endpoint`, or `transmission` |

## 6. Edge attributes

### 6.1 Loaded / directly derived from raw simulation data

| Attribute | Type | Present on | Description |
|---|---|---|---|
| `contact_location` | `tuple[float, float, float]` | all edges | Contact point coordinates |
| `delta` | `float` | all edges | Overlap / displacement quantity from the raw row |
| `delta_t` | `float` | all edges | Tangential displacement quantity from the raw row |
| `normal_force` | `float` | all edges | Normal force magnitude |
| `tangential_force` | `float` | all edges | Tangential force magnitude |
| `n_unit` | `tuple[float, float, float]` | all edges | Stored contact normal direction |
| `t_unit` | `tuple[float, float, float]` | all edges | Stored tangential direction |
| `angle_with_zz` | `float` | all edges | `degrees(arccos(abs(n_z)))` |
| `is_wall_contact` | `bool` | all edges | `True` if either endpoint is a wall node |

### 6.2 Graph-derived edge metrics

These are attached to particle-particle edges in `core`, then mirrored onto the matching full-graph particle-particle edges.

| Attribute | Type | Meaning |
|---|---|---|
| `edge_connectivity` | `int` | Local edge connectivity between the two particle endpoints in core graph |
| `edge_connectivity_with_walls` | `int` | Local edge connectivity between the same two particle endpoints in full graph |
| `node_connectivity` | `int` | Local node connectivity between the two particle endpoints in core graph |
| `node_connectivity_with_walls` | `int` | Local node connectivity between the same two particle endpoints in full graph |
| `curvature_no_walls` | `float` | Ollivier-Ricci curvature from core graph |
| `curvature_with_walls` | `float` | Ollivier-Ricci curvature from full graph, copied onto matching core edges |
| `is_high_force` | `bool` | `True` if `normal_force >= threshold` |

## 7. Graph attributes

Graph attributes are stored on both `G_core.graph` and `G_full.graph`. In most cases both views expose both values:

- `property`
  - core graph result
- `property_with_walls`
  - full graph result

### 7.1 Loaded / bookkeeping graph attributes

| Attribute | Type | Meaning |
|---|---|---|
| `sim_idx` | `int` | Simulation index within one geometry |
| `angle_label` | `str` | One of `0deg`, `15deg`, `30deg`, `45deg` |
| `num_particles` | `int` | Fixed number of particle nodes expected in the slice |
| `num_contacts` | `int` | Number of raw contact rows in the slice |

### 7.2 Size and wall-count summaries

| Attribute | Type | Meaning |
|---|---|---|
| `num_nodes` | `int` | Number of nodes in core graph |
| `num_nodes_with_walls` | `int` | Number of nodes in full graph |
| `num_edges` | `int` | Number of edges in core graph |
| `num_edges_with_walls` | `int` | Number of edges in full graph |
| `wall_nodes` | `int` | Always `0` for core graph |
| `wall_nodes_with_walls` | `int` | Number of wall nodes in full graph |
| `wall_contacts` | `int` | Always `0` for core graph |
| `wall_contacts_with_walls` | `int` | Number of wall-contact edges in full graph |

### 7.3 Connectivity and path summaries

| Attribute | Type | Meaning |
|---|---|---|
| `assortativity` | `float` | Degree assortativity coefficient of core graph |
| `assortativity_with_walls` | `float` | Degree assortativity coefficient of full graph |
| `edge_connectivity_graph` | `int` | Global edge connectivity of core graph |
| `edge_connectivity_graph_with_walls` | `int` | Global edge connectivity of full graph |
| `node_connectivity_graph` | `int` | Global node connectivity of core graph |
| `node_connectivity_graph_with_walls` | `int` | Global node connectivity of full graph |
| `connected` | `bool` | Whether the core graph is connected |
| `connected_with_walls` | `bool` | Whether the full graph is connected |
| `num_components` | `int` | Number of connected components in core graph |
| `num_components_with_walls` | `int` | Number of connected components in full graph |
| `path_nodes` | `int` | Number of nodes in the component used for path metrics in core graph |
| `path_nodes_with_walls` | `int` | Same for full graph |
| `avg_path` | `float` | Average shortest path length on the component used for path metrics |
| `avg_path_with_walls` | `float` | Same for full graph |
| `diameter` | `int or NaN` | Diameter on the component used for path metrics |
| `diameter_with_walls` | `int or NaN` | Same for full graph |
| `radius` | `int or NaN` | Radius on the component used for path metrics |
| `radius_with_walls` | `int or NaN` | Same for full graph |

Important convention:

- If a graph is disconnected, `avg_path`, `diameter`, and `radius` are computed on the largest connected component.
- `connected`, `num_components`, and `path_nodes` tell you exactly how that result should be interpreted.

### 7.4 Loop summaries from the minimum cycle basis

| Attribute | Type | Meaning |
|---|---|---|
| `loop_total` | `int` | Number of basis cycles in the minimum cycle basis |
| `loop_total_with_walls` | `int` | Same for full graph |
| `loop_mean` | `float` | Mean basis-cycle size |
| `loop_mean_with_walls` | `float` | Same for full graph |
| `loop_sizes` | `list[int]` | Sorted list of basis-cycle sizes |
| `loop_sizes_with_walls` | `list[int]` | Same for full graph |
| `loop_counts` | `dict[int, int]` | Mapping from cycle length to count |
| `loop_counts_with_walls` | `dict[int, int]` | Same for full graph |
| `loop_3`, `loop_4`, `loop_5`, ... | `int` | Count of basis cycles of a given size |

Loop convention:

- The script first applies `nx.k_core(G, k=2)` because degree-1 nodes cannot belong to cycles.
- It then runs `nx.minimum_cycle_basis` on that reduced graph.
- Counts are for basis cycles, not for all possible simple cycles.

### 7.5 Adjacency-spectrum attributes

| Attribute | Type | Meaning |
|---|---|---|
| `eigenvalues` | `list[float]` | Largest adjacency eigenvalues for the core graph |
| `eigenvalues_with_walls` | `list[float]` | Largest adjacency eigenvalues for the full graph |
| `eigenvector_nodes` | `list[node_id]` | Node order used for `eigenvectors` |
| `eigenvector_nodes_with_walls` | `list[node_id]` | Node order used for `eigenvectors_with_walls` |
| `eigenvectors` | `list[list[float]]` | Leading adjacency eigenvectors for the core graph |
| `eigenvectors_with_walls` | `list[list[float]]` | Leading adjacency eigenvectors for the full graph |
| `spectral_radius` | `float` | Largest adjacency eigenvalue of core graph |
| `spectral_radius_with_walls` | `float` | Largest adjacency eigenvalue of full graph |

### 7.6 Laplacian-spectrum attributes

| Attribute | Type | Meaning |
|---|---|---|
| `lap_eigenvalues` | `list[float]` | Smallest Laplacian eigenvalues of the core graph |
| `lap_eigenvalues_with_walls` | `list[float]` | Same for full graph |
| `lap_eigenvector_nodes` | `list[node_id]` | Node order used for `lap_eigenvectors` |
| `lap_eigenvector_nodes_with_walls` | `list[node_id]` | Node order used for `lap_eigenvectors_with_walls` |
| `lap_eigenvectors` | `list[list[float]]` | Laplacian eigenvectors in that node order |
| `lap_eigenvectors_with_walls` | `list[list[float]]` | Same for full graph |
| `norm_lap_eigenvalues` | `list[float]` | Smallest normalized Laplacian eigenvalues of the core graph |
| `norm_lap_eigenvalues_with_walls` | `list[float]` | Same for full graph |
| `alg_connectivity` | `float` | Algebraic connectivity, i.e. second-smallest Laplacian eigenvalue |
| `alg_connectivity_with_walls` | `float` | Same for full graph |
| `fiedler_value` | `float` | First strictly positive Laplacian eigenvalue |
| `fiedler_value_with_walls` | `float` | Same for full graph |
| `fiedler_nodes` | `list[node_id]` | Node order used for `fiedler_vector` |
| `fiedler_nodes_with_walls` | `list[node_id]` | Same for full graph |
| `fiedler_vector` | `list[float]` | Fiedler vector for the core graph |
| `fiedler_vector_with_walls` | `list[float]` | Fiedler vector for the full graph |

### 7.7 NFD graph attributes

These come from the unweighted NFD bundle on the core graph and are copied onto both graph views.

| Attribute | Type | Meaning |
|---|---|---|
| `nfd_q` | `list[float]` | Sampled `q` values |
| `nfd_tau` | `list[float]` | `tau(q)` values |
| `nfd_spectrum_q` | `list[float]` | `q` values used for the spectrum slice |
| `nfd_alpha` | `list[float]` | Singularity strengths from the discrete derivative of `tau(q)` |
| `nfd_f_alpha` | `list[float]` | Multifractal spectrum values |
| `nfd_asymmetry` | `float` | Log asymmetry measure of the spectrum |
| `nfd_dimension_q` | `list[float]` | `q` values used for generalized dimensions |
| `nfd_dimension` | `list[float]` | Generalized dimensions `D(q) = tau(q) / q` for `q != 0` |
| `nfd_heat_q` | `list[float]` | `q` values for the discrete specific heat |
| `nfd_heat` | `list[float]` | Discrete specific heat based on the second difference of `tau(q)` |
| `nfd_tau_mean` | `float` | Mean of finite `tau(q)` values |
| `nfd_tau_std` | `float` | Standard deviation of finite `tau(q)` values |

## 8. Pairwise edge-connectivity files

Each file in `pair_edge_connectivity/` contains one row for every unordered particle pair in the core node set.

Columns:

| Column | Type | Meaning |
|---|---|---|
| `geometry` | `str` | Geometry label |
| `sim_idx` | `int` | Simulation index |
| `node1` | `int` | First particle id |
| `node2` | `int` | Second particle id |
| `is_edge` | `bool` | `True` if the pair is an actual core edge |
| `edge_connectivity` | `int` | Pairwise edge connectivity in core graph |
| `edge_connectivity_with_walls` | `int` | Pairwise edge connectivity in full graph |

This file is the answer to the "connectivity for every pair" requirement. The edge attribute table still stores connectivity only on existing edges, while the pair table covers all particle-particle pairs.

## 9. Flat-table conventions

### `node_features.csv`

- One row per particle node from the core graph.
- Coordinates are flattened to `x`, `y`, `z`.
- Any list/dict-like values are JSON strings if they appear.

### `edge_features.csv`

- One row per edge in the full graph.
- `contact_location` is flattened to `contact_x`, `contact_y`, `contact_z`.
- `n_unit` is flattened to `n_x`, `n_y`, `n_z`.
- `t_unit` is flattened to `t_x`, `t_y`, `t_z`.
- `is_core_edge` tells you whether the full-graph edge also exists in the wall-free core graph.

### `graph_features.csv`

- One row per simulation.
- Intended for scalar graph attributes only.
- Best choice when you want simple analysis in pandas without parsing arrays.

### `graph_feature_arrays.csv` and `graph_feature_arrays.pkl`

- Store graph attributes that are naturally array-like or mapping-like.
- CSV version serializes lists/dicts as JSON strings.
- Pickle version preserves native Python objects.

## 10. How the derived properties are computed

This section summarizes the exact implementation choices.

### Centrality and local topology

- `degree`
  - `dict(G.degree())`
- `closeness`
  - `nx.closeness_centrality(G)`
- `betweenness`
  - `nx.betweenness_centrality(G)`
- `clustering`
  - `nx.clustering(G)`
- `avg_neighbor_degree`
  - `nx.average_neighbor_degree(G)`

All of the above are unweighted and computed once on `G_core` and once on `G_full`.

### Graph assortativity coefficient

- `nx.degree_assortativity_coefficient(G)`

This is a graph-level scalar. Node-level average neighbor degree is stored separately as `avg_neighbor_degree`.

### Edge connectivity

- For global graph connectivity:
  - `nx.edge_connectivity(G)`
- For local pairwise edge connectivity:
  - Exact Gomory-Hu trees are built per connected component.
  - Local edge connectivity for a node pair is the minimum cut value along the tree path.
  - If the Gomory-Hu tree cannot be built, the script falls back to `nx.edge_connectivity(G, u, v)`.

This keeps the result exact for undirected graphs while avoiding repeated max-flow solves for every pair.

### Node connectivity

- For global graph connectivity:
  - `nx.node_connectivity(G)`
- For local edge-pair values:
  - `nx.node_connectivity(G, u, v)` on each existing core edge

This is more expensive than edge connectivity, which is why the script already parallelizes it internally.

### Ollivier-Ricci curvature

- Full graph:
  - `OllivierRicci(G_full.copy(), alpha=0.5, verbose="ERROR")`
- Core graph:
  - `OllivierRicci(G_core.copy(), alpha=0.5, verbose="ERROR")`

The script runs curvature twice:

- once with wall contacts included
- once without wall contacts

`avg_curvature_*` is the mean of the incident edge curvatures at each particle node.

### Adjacency spectrum

- Matrix:
  - unweighted adjacency matrix
- Solver:
  - sparse `eigsh(..., which="LA")` when possible
  - dense `np.linalg.eigh` fallback otherwise
- Saved count:
  - `SPECTRAL_NUM_EIGENPAIRS`

`principal_eigenvector` is the first saved adjacency eigenvector component at each node.

### Laplacian spectrum

- Combinatorial Laplacian:
  - `L = D - A`
  - implemented through `scipy.sparse.csgraph.laplacian(..., normed=False)`
- Normalized Laplacian:
  - `scipy.sparse.csgraph.laplacian(..., normed=True)`
- Solver:
  - sparse `eigsh(..., which="SA")` when possible
  - dense `np.linalg.eigh` fallback otherwise

Definitions used by the script:

- `alg_connectivity`
  - second-smallest Laplacian eigenvalue
- `fiedler_value`
  - first strictly positive Laplacian eigenvalue
- `fiedler_vector`
  - eigenvector corresponding to `fiedler_value`

### Path metrics

- `avg_path`
  - `nx.average_shortest_path_length(H)`
- `diameter`
  - `nx.diameter(H)`
- `radius`
  - `nx.radius(H)`

Here `H` is:

- the whole graph if the graph is connected
- otherwise the largest connected component

### Loop metrics

- The script first computes `nx.k_core(G, k=2)`.
- It then runs `nx.minimum_cycle_basis(H)`.
- It records:
  - the total number of basis cycles
  - mean basis-cycle size
  - the full sorted list of basis-cycle sizes
  - counts by basis-cycle size, exposed as both `loop_counts` and scalar keys such as `loop_3`, `loop_4`, `loop_5`, ...

### NFD

At node level:

- For each node, shortest-path shells are computed with `nx.single_source_shortest_path_length`.
- The cumulative shell count versus radius is fitted in log-log space.
- The slope is stored as `nfd`.
- The fit quality is stored as `nfd_r2`.

At graph level:

- The shell-fraction distributions across nodes are aggregated over `q`.
- The script computes `tau(q)`, spectrum quantities, generalized dimensions, and discrete heat-like summaries.

### High-force labels

The threshold is set from one of three modes:

- reference geometry
- global
- per geometry

If `quantile_threshold` is `None`, the threshold is:

- `2 * mean(normal_force)`

Then:

- `is_high_force = normal_force >= threshold`
- `high_force_degree`
  - number of incident high-force edges in the full graph
- `force_chain_role`
  - `none` if degree is `0`
  - `endpoint` if degree is `1`
  - `transmission` if degree is `>= 2`

## 11. Parallelization now used by the script

The script now uses two parallel layers, but not at full strength at the same time:

- simulation-level parallelism
  - each simulation slice is processed independently
  - enabled by `RUN_SIM_PARALLEL`
- edge-level node-connectivity parallelism
  - still used inside `compute_dual_view_metrics`
  - automatically reduced to one worker when simulation-level parallelism is enabled

This avoids nested-process oversubscription while preserving exactly the same calculated values as the serial version.

Relevant settings:

- `RUN_SIM_PARALLEL`
- `SIM_N_JOBS`
- `SIM_PARALLEL_VERBOSE`
- `NODE_CONN_N_JOBS`
- `NODE_CONN_N_JOBS_WHEN_SIM_PARALLEL`
- `NODE_CONN_VERBOSE`

Recommended interpretation:

- If the machine has enough memory and many independent simulation slices, keep `RUN_SIM_PARALLEL=True`.
- If you want the original behavior inside each simulation and only one simulation at a time, set `RUN_SIM_PARALLEL=False`.
