import os
import csv
import json
import re
import numpy as np
import scipy.io
import networkx as nx
import pickle
import pandas as pd
from collections import defaultdict, Counter
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.sparse import csgraph
from scipy.stats import linregress
from scipy.sparse.linalg import eigsh

# === Config ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
base_path = os.path.join(PROJECT_ROOT, 'Data', 'PeriodicBoudaries')

def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"⚠️ Invalid integer for {name}={value!r}; using default {default}.")
        return default


out_path = os.environ.get(
    'GRAPHGEN_OUT_PATH',
    os.path.join(PROJECT_ROOT, 'AnalysisResults', 'PeriodicBoudaries', 'GraphGeneration'),
)
os.makedirs(out_path, exist_ok=True)
os.makedirs(os.path.join(out_path, 'pair_edge_connectivity'), exist_ok=True)


# Threshold mode toggles:
USE_GLOBAL_HIGH_FORCE_THRESHOLD = False  # True → single threshold across all geometries (unless reference mode set)
REFERENCE_GEOMETRY_LABEL = '0deg'       # e.g., '0deg' to lock threshold to 0°, or set to None to disable reference mode
USE_CORE_FOR_THRESHOLD = False          # True → compute threshold (and plot) from CORE graphs (no wall contacts)

# Plot settings
PLOT_FORCE_DISTRIBUTIONS = True
FORCE_DIST_BINS = 100
FORCE_DIST_LOGX = False  # set True if forces span orders of magnitude

# Layout & style for force-distribution figure
FORCE_DIST_COLUMN = True          # True => Nx1 column; False => 1xN row
HIST_LINE_WIDTH = 1.0             # outline width for the connected histogram
THRESHOLD_LINE_WIDTH = 1.0        # a bit thinner than before
THRESHOLD_LINE_COLOR = 'red'      # red dashed threshold line

# NFD settings
RUN_NFD_ANALYSIS = True
NFD_Q_VALUES = [q / 100 for q in range(-2000, 2001, 10)]
NFD_SPECTRUM_Q_WINDOW = (-3.0, 3.0)

# Spectral settings (adjacency + Laplacian eigensystems)
RUN_SPECTRAL_ANALYSIS = True
SPECTRAL_NUM_EIGENPAIRS = 3

# Parallel settings
# Simulation-level parallelism is the safest acceleration path because each graph is independent.
RUN_SIM_PARALLEL = _env_flag('GRAPHGEN_RUN_SIM_PARALLEL', True)
SIM_N_JOBS = max(1, _env_int('GRAPHGEN_SIM_N_JOBS', max(1, (os.cpu_count() or 1) - 1)))
SIM_PARALLEL_VERBOSE = 10
# When simulation-level parallelism is enabled, keep inner node-connectivity parallelism at 1
# to avoid nested-process oversubscription. Set RUN_SIM_PARALLEL=False to use full inner parallelism.
NODE_CONN_N_JOBS = _env_int('GRAPHGEN_NODE_CONN_N_JOBS', -1)
NODE_CONN_N_JOBS_WHEN_SIM_PARALLEL = _env_int('GRAPHGEN_NODE_CONN_N_JOBS_WHEN_SIM_PARALLEL', 1)
NODE_CONN_VERBOSE = _env_int('GRAPHGEN_NODE_CONN_VERBOSE', 0)


# === Data containers ===
graph_dict = {}
slice_info = []
pair_edge_index_records = []


def _safe_degree_assortativity_coefficient(G):
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return np.nan
    try:
        return float(nx.degree_assortativity_coefficient(G))
    except Exception:
        return np.nan


def _safe_graph_connectivities(G):
    if G.number_of_nodes() < 2:
        return 0, 0
    try:
        edge_conn = int(nx.edge_connectivity(G))
    except Exception:
        edge_conn = 0
    try:
        node_conn = int(nx.node_connectivity(G))
    except Exception:
        node_conn = 0
    return edge_conn, node_conn


def _set_dual_graph_attr(G_core, G_full, key, core_value, full_value):
    G_core.graph[key] = core_value
    G_core.graph[f'{key}_with_walls'] = full_value
    G_full.graph[key] = core_value
    G_full.graph[f'{key}_with_walls'] = full_value


def _standardize_eigenpairs(eigvals, eigvecs, descending):
    eigvals = np.real_if_close(eigvals)
    eigvecs = np.real_if_close(eigvecs)

    order = np.argsort(eigvals)
    if descending:
        order = order[::-1]

    eigvals = np.asarray(eigvals[order], dtype=float)
    eigvecs = np.asarray(eigvecs[:, order], dtype=float)

    for j in range(eigvecs.shape[1]):
        pivot = int(np.argmax(np.abs(eigvecs[:, j])))
        if eigvecs[pivot, j] < 0:
            eigvecs[:, j] *= -1.0

    return eigvals, eigvecs


def _compute_matrix_eigenpairs(matrix, num_eigenpairs=3, which='LA', descending=True):
    n_nodes = matrix.shape[0]

    if n_nodes == 0:
        return np.array([], dtype=float), np.zeros((0, 0), dtype=float)

    if n_nodes == 1:
        value = float(matrix.toarray()[0, 0]) if hasattr(matrix, 'toarray') else float(matrix[0, 0])
        return np.array([value], dtype=float), np.array([[1.0]], dtype=float)

    k = int(max(1, min(num_eigenpairs, n_nodes - 1)))

    try:
        eigvals, eigvecs = eigsh(matrix, k=k, which=which)
    except Exception:
        dense = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix, dtype=float)
        eigvals, eigvecs = np.linalg.eigh(dense)

    eigvals, eigvecs = _standardize_eigenpairs(eigvals, eigvecs, descending=descending)
    return eigvals[:k], eigvecs[:, :k]


def _compute_unweighted_nfd_bundle(G, q_values, spectrum_q_window=(-3.0, 3.0)):
    """
    NMFA-style unweighted NFD bundle:
      - node-level NFD slope and fit quality
      - graph-level tau(q), spectrum (alpha, f(alpha), asymmetry), D(q), heat
    """
    q_array = np.asarray(q_values, dtype=float)
    node_metrics = {}
    shell_fraction_lists = []
    rmax_list = []

    for node in G.nodes():
        spl = nx.single_source_shortest_path_length(G, node)
        grow = [s for s in spl.values() if s > 0]

        if not grow:
            node_metrics[node] = {
                'nfd': np.nan,
                'nfd_r2': np.nan,
            }
            continue

        grow.sort()
        counts_by_radius = Counter(grow)
        total_reachable = sum(counts_by_radius.values())

        radii = []
        cumulative_counts = []
        cumulative_fractions = []
        cumulative = 0

        for radius, count in sorted(counts_by_radius.items()):
            cumulative += count
            if radius > 0:
                radii.append(float(radius))
                cumulative_counts.append(float(cumulative))
                cumulative_fractions.append(float(cumulative) / float(total_reachable))

        if len(radii) >= 2 and len(cumulative_counts) >= 2:
            x = np.log(np.asarray(radii, dtype=float))
            y = np.log(np.asarray(cumulative_counts, dtype=float))
            slope, _, r_value, _, _ = linregress(x, y)
            node_metrics[node] = {
                'nfd': float(slope),
                'nfd_r2': float(r_value ** 2),
            }
        else:
            node_metrics[node] = {
                'nfd': np.nan,
                'nfd_r2': np.nan,
            }

        if cumulative_fractions:
            shell_fraction_lists.append(cumulative_fractions)
            rmax_list.append(max(counts_by_radius))

    tau_values = np.full(len(q_array), np.nan, dtype=float)
    if shell_fraction_lists and rmax_list:
        diameter = int(np.max(rmax_list))
        if diameter >= 2:
            frac_mat = np.ones((len(shell_fraction_lists), diameter), dtype=float)
            for ridx, shell_fracs in enumerate(shell_fraction_lists):
                sf = np.asarray(shell_fracs, dtype=float)
                frac_mat[ridx, :sf.size] = sf

            x = np.log(np.arange(1, diameter + 1, dtype=float) / float(diameter))
            for idx in range(len(q_array)):
                y_raw = np.sum(np.power(frac_mat, q_array[idx]), axis=0)
                y = np.log(y_raw)
                valid = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(valid) >= 2:
                    slope, _, _, _, _ = linregress(x[valid], y[valid])
                    tau_values[idx] = float(slope)

    # Spectrum in a bounded q-window (same spirit as NMFA's central slice)
    q_min, q_max = spectrum_q_window
    mask_window = (q_array >= q_min) & (q_array <= q_max) & np.isfinite(tau_values)
    q_window = q_array[mask_window]
    tau_window = tau_values[mask_window]

    alpha_values = np.array([], dtype=float)
    f_alpha_values = np.array([], dtype=float)
    q_spectrum = np.array([], dtype=float)
    asymmetry = np.nan

    if len(q_window) >= 3:
        dq = np.diff(q_window)
        dtau = np.diff(tau_window)
        valid_grad = dq != 0
        if np.any(valid_grad):
            alpha_values = dtau[valid_grad] / dq[valid_grad]
            q_spectrum = q_window[:-1][valid_grad]
            tau_for_f = tau_window[:-1][valid_grad]
            f_alpha_values = q_spectrum * alpha_values - tau_for_f

            if len(alpha_values) >= 3 and len(f_alpha_values) == len(alpha_values):
                peak_idx = int(np.argmax(f_alpha_values))
                left_span = alpha_values[peak_idx] - alpha_values[0]
                right_span = alpha_values[-1] - alpha_values[peak_idx]
                if left_span > 0 and right_span > 0:
                    asymmetry = float(np.log(right_span / left_span))

    # Generalized dimension D(q) = tau(q)/q for q != 0
    dim_mask = (q_array != 0) & np.isfinite(tau_values)
    q_dim = q_array[dim_mask]
    dim_values = tau_values[dim_mask] / q_dim

    # Specific heat (discrete second derivative of tau)
    heat_q_values = []
    heat_values = []
    for i in range(2, len(tau_values)):
        t0, t1, t2 = tau_values[i - 2], tau_values[i - 1], tau_values[i]
        if np.isfinite(t0) and np.isfinite(t1) and np.isfinite(t2):
            heat_values.append(float(-100.0 * (t2 - 2.0 * t1 + t0)))
            heat_q_values.append(float(q_array[i]))

    finite_tau = tau_values[np.isfinite(tau_values)]
    tau_mean = float(np.mean(finite_tau)) if finite_tau.size else np.nan
    tau_std = float(np.std(finite_tau)) if finite_tau.size else np.nan

    graph_metrics = {
        'nfd_q': q_array.tolist(),
        'nfd_tau': tau_values.tolist(),
        'nfd_spectrum_q': q_spectrum.tolist(),
        'nfd_alpha': alpha_values.tolist(),
        'nfd_f_alpha': f_alpha_values.tolist(),
        'nfd_asymmetry': float(asymmetry) if np.isfinite(asymmetry) else np.nan,
        'nfd_dimension_q': q_dim.tolist(),
        'nfd_dimension': dim_values.tolist(),
        'nfd_heat_q': heat_q_values,
        'nfd_heat': heat_values,
        'nfd_tau_mean': tau_mean,
        'nfd_tau_std': tau_std,
    }

    return node_metrics, graph_metrics


def _compute_adjacency_eigenpairs(G, num_eigenpairs=3):
    """
    Compute largest adjacency eigenpairs for an undirected graph.
    Returns
    -------
    node_order : list
    eigenvalues_desc : np.ndarray shape (k,)
    eigenvectors_desc : np.ndarray shape (n_nodes, k)
    """
    node_order = list(G.nodes())
    n_nodes = len(node_order)

    if n_nodes == 0:
        return node_order, np.array([], dtype=float), np.zeros((0, 0), dtype=float)

    A = nx.to_scipy_sparse_array(G, nodelist=node_order, dtype=float, weight=None, format='csr')
    eigvals, eigvecs = _compute_matrix_eigenpairs(
        A,
        num_eigenpairs=num_eigenpairs,
        which='LA',
        descending=True,
    )
    return node_order, eigvals, eigvecs


def _compute_laplacian_bundle(G, num_eigenpairs=3, zero_tol=1e-9):
    node_order = list(G.nodes())
    n_nodes = len(node_order)

    if n_nodes == 0:
        return {
            'node_order': node_order,
            'lap_eigenvalues': np.array([], dtype=float),
            'lap_eigenvectors': np.zeros((0, 0), dtype=float),
            'norm_lap_eigenvalues': np.array([], dtype=float),
            'alg_connectivity': np.nan,
            'fiedler_value': np.nan,
            'fiedler_vector': np.array([], dtype=float),
        }

    if n_nodes == 1:
        return {
            'node_order': node_order,
            'lap_eigenvalues': np.array([0.0], dtype=float),
            'lap_eigenvectors': np.array([[1.0]], dtype=float),
            'norm_lap_eigenvalues': np.array([0.0], dtype=float),
            'alg_connectivity': 0.0,
            'fiedler_value': np.nan,
            'fiedler_vector': np.array([np.nan], dtype=float),
        }

    adjacency = nx.to_scipy_sparse_array(G, nodelist=node_order, dtype=float, weight=None, format='csr')
    n_components = nx.number_connected_components(G)
    target_k = int(max(1, min(n_nodes - 1, max(num_eigenpairs, n_components + 1))))

    laplacian = csgraph.laplacian(adjacency, normed=False)
    norm_laplacian = csgraph.laplacian(adjacency, normed=True)

    lap_eigvals, lap_eigvecs = _compute_matrix_eigenpairs(
        laplacian,
        num_eigenpairs=target_k,
        which='SA',
        descending=False,
    )
    norm_lap_eigvals, _ = _compute_matrix_eigenpairs(
        norm_laplacian,
        num_eigenpairs=target_k,
        which='SA',
        descending=False,
    )

    alg_connectivity = float(lap_eigvals[1]) if lap_eigvals.size > 1 else 0.0
    positive_idx = np.where(lap_eigvals > zero_tol)[0]
    if positive_idx.size:
        fiedler_idx = int(positive_idx[0])
        fiedler_value = float(lap_eigvals[fiedler_idx])
        fiedler_vector = np.asarray(lap_eigvecs[:, fiedler_idx], dtype=float)
    else:
        fiedler_value = np.nan
        fiedler_vector = np.full(n_nodes, np.nan, dtype=float)

    return {
        'node_order': node_order,
        'lap_eigenvalues': lap_eigvals,
        'lap_eigenvectors': lap_eigvecs,
        'norm_lap_eigenvalues': norm_lap_eigvals,
        'alg_connectivity': alg_connectivity,
        'fiedler_value': fiedler_value,
        'fiedler_vector': fiedler_vector,
    }


def _local_edge_conn_from_gh_tree(T, u, v):
    if u == v:
        return 0
    path = nx.shortest_path(T, u, v)
    return int(min(T[a][b]['weight'] for a, b in zip(path[:-1], path[1:])))


def _build_gomory_hu_map(G):
    tree_map = {}
    if G.number_of_nodes() < 2:
        return tree_map

    for nodes in nx.connected_components(G):
        nodes = list(nodes)
        if len(nodes) < 2:
            tree = None
        else:
            try:
                tree = nx.gomory_hu_tree(G.subgraph(nodes).copy())
            except Exception:
                tree = None
        for node in nodes:
            tree_map[node] = tree

    return tree_map


def _local_edge_connectivity(G, gh_map, u, v):
    if u == v:
        return 0

    tree_u = gh_map.get(u) if gh_map else None
    tree_v = gh_map.get(v) if gh_map else None
    if tree_u is not None and tree_u is tree_v:
        return _local_edge_conn_from_gh_tree(tree_u, u, v)

    try:
        if not nx.has_path(G, u, v):
            return 0
    except Exception:
        return 0

    try:
        return int(nx.edge_connectivity(G, u, v))
    except Exception:
        return 0


def _compute_graph_distance_metrics(G):
    if G.number_of_nodes() == 0:
        return {
            'connected': False,
            'num_components': 0,
            'path_nodes': 0,
            'avg_path': np.nan,
            'diameter': np.nan,
            'radius': np.nan,
        }

    num_components = nx.number_connected_components(G)
    connected = (num_components == 1)

    if connected:
        H = G
    else:
        largest_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_nodes)

    path_nodes = H.number_of_nodes()
    if path_nodes == 1:
        avg_path = 0.0
        diameter = 0
        radius = 0
    else:
        avg_path = float(nx.average_shortest_path_length(H))
        diameter = int(nx.diameter(H))
        radius = int(nx.radius(H))

    return {
        'connected': bool(connected),
        'num_components': int(num_components),
        'path_nodes': int(path_nodes),
        'avg_path': avg_path,
        'diameter': diameter,
        'radius': radius,
    }


def _compute_loop_metrics(G):
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return {
            'loop_total': 0,
            'loop_mean': np.nan,
            'loop_sizes': [],
            'loop_counts': {},
        }

    H = nx.k_core(G, k=2)
    if H.number_of_nodes() < 3 or H.number_of_edges() == 0:
        return {
            'loop_total': 0,
            'loop_mean': np.nan,
            'loop_sizes': [],
            'loop_counts': {},
        }

    loop_sizes = sorted(len(cycle) for cycle in nx.minimum_cycle_basis(H) if len(cycle) >= 3)
    counts = dict(sorted(Counter(loop_sizes).items()))
    metrics = {
        'loop_total': int(len(loop_sizes)),
        'loop_mean': float(np.mean(loop_sizes)) if loop_sizes else np.nan,
        'loop_sizes': loop_sizes,
        'loop_counts': counts,
    }
    for size, count in counts.items():
        metrics[f'loop_{size}'] = int(count)
    return metrics


def _iter_pair_edge_connectivity_records(
    G_core,
    G_full,
    gh_core_map,
    gh_full_map,
):
    node_list = list(G_core.nodes())
    graph_id = {
        'geometry': G_full.graph.get('angle_label'),
        'sim_idx': G_full.graph.get('sim_idx'),
    }

    for idx, u in enumerate(node_list[:-1]):
        for v in node_list[idx + 1:]:
            yield {
                **graph_id,
                'node1': u,
                'node2': v,
                'is_edge': bool(G_core.has_edge(u, v)),
                'edge_connectivity': _local_edge_connectivity(G_core, gh_core_map, u, v),
                'edge_connectivity_with_walls': _local_edge_connectivity(G_full, gh_full_map, u, v),
            }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _serialize_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), default=_json_default)
    if isinstance(value, tuple):
        return json.dumps(list(value), default=_json_default)
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=_json_default)
    return value


def _flatten_named_vector(record, key, value, out_names):
    if value is None:
        for name in out_names:
            record[name] = np.nan
        return True

    try:
        flat = np.asarray(value, dtype=object).reshape(-1)
    except Exception:
        return False

    if flat.size != len(out_names):
        return False

    for name, item in zip(out_names, flat):
        record[name] = item.item() if isinstance(item, np.generic) else item
    return True

# === Helper to compute dual metrics ===
def compute_dual_view_metrics(G_full, node_conn_n_jobs=-1, node_conn_verbose=1, pair_edge_out_dir=None):
    """
    Returns
    -------
    G_core : nx.Graph
        Core graph (no wall nodes) with node/edge/graph metrics attached.
    pair_edge_index_entry : dict | None
        Metadata for the per-graph all-pairs particle table written to disk.
    """
    core_nodes = [n for n, d in G_full.nodes(data=True) if not d.get('is_wall', False)]
    G_core = G_full.subgraph(core_nodes).copy()
    wall_nodes_with_walls = sum(1 for _, d in G_full.nodes(data=True) if d.get('is_wall', False))
    wall_contacts_with_walls = sum(1 for _, _, d in G_full.edges(data=True) if d.get('is_wall_contact', False))

    _set_dual_graph_attr(G_core, G_full, 'num_nodes', G_core.number_of_nodes(), G_full.number_of_nodes())
    _set_dual_graph_attr(G_core, G_full, 'num_edges', G_core.number_of_edges(), G_full.number_of_edges())
    _set_dual_graph_attr(G_core, G_full, 'wall_nodes', 0, wall_nodes_with_walls)
    _set_dual_graph_attr(G_core, G_full, 'wall_contacts', 0, wall_contacts_with_walls)

    # Core-only centralities
    degree_core = dict(G_core.degree())
    closeness_core = nx.closeness_centrality(G_core)
    betweenness_core = nx.betweenness_centrality(G_core)
    clustering_core = nx.clustering(G_core)

    # Full-graph centralities
    degree_full = dict(G_full.degree())
    closeness_full = nx.closeness_centrality(G_full)
    betweenness_full = nx.betweenness_centrality(G_full)
    clustering_full = nx.clustering(G_full)
    avg_neighbor_degree_core = nx.average_neighbor_degree(G_core)
    avg_neighbor_degree_full = nx.average_neighbor_degree(G_full)

    # Graph-level assortativity coefficients and connectivity (single values per graph)
    degree_assortativity_no_walls = _safe_degree_assortativity_coefficient(G_core)
    degree_assortativity_with_walls = _safe_degree_assortativity_coefficient(G_full)
    edge_conn_no_walls, node_conn_no_walls = _safe_graph_connectivities(G_core)
    edge_conn_with_walls, node_conn_with_walls = _safe_graph_connectivities(G_full)

    _set_dual_graph_attr(G_core, G_full, 'assortativity', degree_assortativity_no_walls, degree_assortativity_with_walls)
    _set_dual_graph_attr(G_core, G_full, 'edge_connectivity_graph', edge_conn_no_walls, edge_conn_with_walls)
    _set_dual_graph_attr(G_core, G_full, 'node_connectivity_graph', node_conn_no_walls, node_conn_with_walls)

    core_distance_metrics = _compute_graph_distance_metrics(G_core)
    full_distance_metrics = _compute_graph_distance_metrics(G_full)
    for key in core_distance_metrics:
        _set_dual_graph_attr(G_core, G_full, key, core_distance_metrics[key], full_distance_metrics[key])

    core_loop_metrics = _compute_loop_metrics(G_core)
    full_loop_metrics = _compute_loop_metrics(G_full)
    loop_keys = sorted(set(core_loop_metrics) | set(full_loop_metrics))
    for key in loop_keys:
        if key.startswith('loop_') and key not in {'loop_total', 'loop_mean', 'loop_sizes', 'loop_counts'}:
            default = 0
        elif key == 'loop_sizes':
            default = []
        elif key == 'loop_counts':
            default = {}
        else:
            default = np.nan
        _set_dual_graph_attr(
            G_core,
            G_full,
            key,
            core_loop_metrics.get(key, default),
            full_loop_metrics.get(key, default),
        )

    if RUN_SPECTRAL_ANALYSIS:
        core_spec_nodes, core_eigvals, core_eigvecs = _compute_adjacency_eigenpairs(
            G_core,
            num_eigenpairs=SPECTRAL_NUM_EIGENPAIRS,
        )

        full_spec_nodes, full_eigvals, full_eigvecs = _compute_adjacency_eigenpairs(
            G_full,
            num_eigenpairs=SPECTRAL_NUM_EIGENPAIRS,
        )
        _set_dual_graph_attr(G_core, G_full, 'eigenvalues', core_eigvals.tolist(), full_eigvals.tolist())
        _set_dual_graph_attr(G_core, G_full, 'eigenvector_nodes', core_spec_nodes, full_spec_nodes)
        _set_dual_graph_attr(G_core, G_full, 'eigenvectors', core_eigvecs.tolist(), full_eigvecs.tolist())
        _set_dual_graph_attr(
            G_core,
            G_full,
            'spectral_radius',
            float(core_eigvals[0]) if core_eigvals.size else np.nan,
            float(full_eigvals[0]) if full_eigvals.size else np.nan,
        )

        core_lap_bundle = _compute_laplacian_bundle(G_core, num_eigenpairs=SPECTRAL_NUM_EIGENPAIRS)
        full_lap_bundle = _compute_laplacian_bundle(G_full, num_eigenpairs=SPECTRAL_NUM_EIGENPAIRS)
        _set_dual_graph_attr(
            G_core,
            G_full,
            'lap_eigenvalues',
            core_lap_bundle['lap_eigenvalues'].tolist(),
            full_lap_bundle['lap_eigenvalues'].tolist(),
        )
        _set_dual_graph_attr(
            G_core,
            G_full,
            'lap_eigenvector_nodes',
            core_lap_bundle['node_order'],
            full_lap_bundle['node_order'],
        )
        _set_dual_graph_attr(
            G_core,
            G_full,
            'lap_eigenvectors',
            core_lap_bundle['lap_eigenvectors'].tolist(),
            full_lap_bundle['lap_eigenvectors'].tolist(),
        )
        _set_dual_graph_attr(
            G_core,
            G_full,
            'norm_lap_eigenvalues',
            core_lap_bundle['norm_lap_eigenvalues'].tolist(),
            full_lap_bundle['norm_lap_eigenvalues'].tolist(),
        )
        _set_dual_graph_attr(
            G_core,
            G_full,
            'alg_connectivity',
            core_lap_bundle['alg_connectivity'],
            full_lap_bundle['alg_connectivity'],
        )
        _set_dual_graph_attr(
            G_core,
            G_full,
            'fiedler_value',
            core_lap_bundle['fiedler_value'],
            full_lap_bundle['fiedler_value'],
        )
        _set_dual_graph_attr(
            G_core,
            G_full,
            'fiedler_nodes',
            core_lap_bundle['node_order'],
            full_lap_bundle['node_order'],
        )
        _set_dual_graph_attr(
            G_core,
            G_full,
            'fiedler_vector',
            core_lap_bundle['fiedler_vector'].tolist(),
            full_lap_bundle['fiedler_vector'].tolist(),
        )

        if core_eigvecs.size:
            core_principal = core_eigvecs[:, 0]
            for idx, node in enumerate(core_spec_nodes):
                G_core.nodes[node]['principal_eigenvector'] = float(core_principal[idx])

        if full_eigvecs.size:
            full_principal = full_eigvecs[:, 0]
            for idx, node in enumerate(full_spec_nodes):
                G_full.nodes[node]['principal_eigenvector_with_walls'] = float(full_principal[idx])

        for idx, node in enumerate(core_lap_bundle['node_order']):
            value = core_lap_bundle['fiedler_vector'][idx]
            G_core.nodes[node]['fiedler'] = float(value) if np.isfinite(value) else np.nan

        for idx, node in enumerate(full_lap_bundle['node_order']):
            value = full_lap_bundle['fiedler_vector'][idx]
            G_full.nodes[node]['fiedler_with_walls'] = float(value) if np.isfinite(value) else np.nan

    edges_list = list(G_core.edges())

    gh_core_map = _build_gomory_hu_map(G_core)
    gh_full_map = _build_gomory_hu_map(G_full)
    pair_edge_index_entry = None
    if pair_edge_out_dir is not None:
        pair_edge_index_entry = _save_pair_edge_connectivity_records(
            _iter_pair_edge_connectivity_records(
                G_core,
                G_full,
                gh_core_map,
                gh_full_map,
            ),
            out_dir=pair_edge_out_dir,
            geometry=G_full.graph.get('angle_label'),
            sim_idx=G_full.graph.get('sim_idx'),
        )

    for u, v in edges_list:
        edge_conn_core = _local_edge_connectivity(G_core, gh_core_map, u, v)
        edge_conn_full = _local_edge_connectivity(G_full, gh_full_map, u, v)
        G_core[u][v]['edge_connectivity'] = edge_conn_core
        G_core[u][v]['edge_connectivity_with_walls'] = edge_conn_full

    # Node connectivity (NP-hard; use parallel processing for speedup)
    def _compute_node_connectivity_pair(u, v, G):
        return (u, v, nx.node_connectivity(G, u, v))

    # Parallel: node connectivity (no walls) for core edges
    print(f"Computing node connectivity (core, {len(edges_list)} edges) with parallel processing...")
    if edges_list:
        results_core = Parallel(n_jobs=node_conn_n_jobs, verbose=node_conn_verbose)(
            delayed(_compute_node_connectivity_pair)(u, v, G_core) for u, v in edges_list
        )
    else:
        results_core = []
    for u, v, nc in results_core:
        G_core[u][v]['node_connectivity'] = nc

    # Parallel: node connectivity (with walls) for core edges
    print(f"Computing node connectivity (full, {len(edges_list)} edges) with parallel processing...")
    if edges_list:
        results_full = Parallel(n_jobs=node_conn_n_jobs, verbose=node_conn_verbose)(
            delayed(_compute_node_connectivity_pair)(u, v, G_full) for u, v in edges_list
        )
    else:
        results_full = []
    for u, v, nc in results_full:
        G_core[u][v]['node_connectivity_with_walls'] = nc

    # Assign dual-view node metrics (only to core nodes)
    for node in G_core.nodes():
        G_core.nodes[node]['degree'] = degree_core.get(node, 0)
        G_core.nodes[node]['closeness'] = closeness_core.get(node, 0.0)
        G_core.nodes[node]['betweenness'] = betweenness_core.get(node, 0.0)
        G_core.nodes[node]['clustering'] = clustering_core.get(node, 0.0)
        G_core.nodes[node]['avg_neighbor_degree'] = avg_neighbor_degree_core.get(node, 0.0)

        G_core.nodes[node]['degree_with_walls'] = degree_full.get(node, 0)
        G_core.nodes[node]['closeness_with_walls'] = closeness_full.get(node, 0.0)
        G_core.nodes[node]['betweenness_with_walls'] = betweenness_full.get(node, 0.0)
        G_core.nodes[node]['clustering_with_walls'] = clustering_full.get(node, 0.0)
        G_core.nodes[node]['avg_neighbor_degree_with_walls'] = avg_neighbor_degree_full.get(node, 0.0)

    # Edge curvature (with walls)
    orc_full = OllivierRicci(G_full.copy(), alpha=0.5, verbose="ERROR")
    orc_full.compute_ricci_curvature()
    for u, v, d in orc_full.G.edges(data=True):
        if u in G_core.nodes and v in G_core.nodes and G_core.has_edge(u, v):
            G_core[u][v]['curvature_with_walls'] = d.get('ricciCurvature', 0.0)

    # Edge curvature (no walls)
    orc_core = OllivierRicci(G_core.copy(), alpha=0.5, verbose="ERROR")
    orc_core.compute_ricci_curvature()
    for u, v, d in orc_core.G.edges(data=True):
        G_core[u][v]['curvature_no_walls'] = d.get('ricciCurvature', 0.0)

    # Average node curvature (both views)
    def assign_avg_curv(G, edge_key, node_key):
        curv_sum = defaultdict(float)
        count = defaultdict(int)
        for a, b, ed in G.edges(data=True):
            if edge_key in ed:
                curv_sum[a] += ed[edge_key]
                curv_sum[b] += ed[edge_key]
                count[a] += 1
                count[b] += 1
        for n in G.nodes:
            G.nodes[n][node_key] = curv_sum[n] / count[n] if count[n] > 0 else 0.0

    assign_avg_curv(G_core, 'curvature_with_walls', 'avg_curvature_with_walls')
    assign_avg_curv(G_core, 'curvature_no_walls', 'avg_curvature_no_walls')

    if RUN_NFD_ANALYSIS:
        print(f"Computing unweighted NFD metrics on core graph (nodes={G_core.number_of_nodes()}, edges={G_core.number_of_edges()})...")
        nfd_node_metrics, nfd_graph_metrics = _compute_unweighted_nfd_bundle(
            G_core,
            q_values=NFD_Q_VALUES,
            spectrum_q_window=NFD_SPECTRUM_Q_WINDOW,
        )

        for node, vals in nfd_node_metrics.items():
            if node in G_core.nodes:
                G_core.nodes[node].update(vals)

        G_core.graph.update(nfd_graph_metrics)
        G_full.graph.update(nfd_graph_metrics)

    for node in G_core.nodes():
        if node in G_full.nodes:
            G_core.nodes[node]['principal_eigenvector_with_walls'] = G_full.nodes[node].get(
                'principal_eigenvector_with_walls', np.nan
            )
            G_core.nodes[node]['fiedler_with_walls'] = G_full.nodes[node].get(
                'fiedler_with_walls', np.nan
            )

    # Push metrics back onto the full graph for core nodes/edges
    for node in G_core.nodes:
        if node in G_full.nodes:
            G_full.nodes[node].update(G_core.nodes[node])
    for a, b in G_core.edges:
        if G_full.has_edge(a, b):
            G_full[a][b].update(G_core[a][b])

    return G_core, pair_edge_index_entry

# === High-force tagging core ===
def _tag_high_force_edges_and_nodes(G_full, G_core, threshold):
    edge_records = []
    # Mark high-force edges in FULL graph
    for u, v, d in G_full.edges(data=True):
        d['is_high_force'] = d.get('normal_force', 0.0) >= threshold

    # Accumulate high-force degrees for non-wall nodes
    high_force_deg = defaultdict(int)
    for u, v, d in G_full.edges(data=True):
        if d['is_high_force']:
            if not G_full.nodes[u].get('is_wall', False):
                high_force_deg[u] += 1
            if not G_full.nodes[v].get('is_wall', False):
                high_force_deg[v] += 1

            def map_node(n):
                return n if not G_full.nodes[n].get('is_wall', False) else -1

            edge_records.append({
                'geometry': G_full.graph.get('angle_label'),
                'sim_idx': G_full.graph.get('sim_idx'),
                'node1': map_node(u),
                'node2': map_node(v)
            })

    # Node roles (FULL graph → for non-wall nodes)
    for n in G_full.nodes():
        if not G_full.nodes[n].get('is_wall', False):
            deg = high_force_deg.get(n, 0)
            G_full.nodes[n]['high_force_degree'] = deg
            G_full.nodes[n]['is_force_chain_node'] = deg >= 1
            if deg == 0:
                G_full.nodes[n]['force_chain_role'] = 'none'
            elif deg == 1:
                G_full.nodes[n]['force_chain_role'] = 'endpoint'
            else:
                G_full.nodes[n]['force_chain_role'] = 'transmission'

    # Mirror edge flag and node roles to CORE graph
    for u, v in G_core.edges():
        if G_full.has_edge(u, v):
            G_core[u][v]['is_high_force'] = G_full[u][v].get('is_high_force', False)
    for n in G_core.nodes():
        deg = high_force_deg.get(n, 0)
        G_core.nodes[n]['high_force_degree'] = deg
        G_core.nodes[n]['is_force_chain_node'] = deg >= 1
        if deg == 0:
            G_core.nodes[n]['force_chain_role'] = 'none'
        elif deg == 1:
            G_core.nodes[n]['force_chain_role'] = 'endpoint'
        else:
            G_core.nodes[n]['force_chain_role'] = 'transmission'

    return edge_records

# === UPDATED: Labeling logic with explicit if/elif/else ===
def label_high_force_edges(
    graph_dict,
    use_global=True,
    quantile_threshold=None,
    use_core_for_threshold=False,
    reference_geometry_label=None,
):
    """
    Labels high-force edges and assigns force-chain node roles.

    Mode precedence (mutually exclusive):
      1) reference_geometry_label (e.g., '0deg') → threshold from that geometry ONLY, applied to all.
      2) elif use_global → single global threshold across all geometries.
      3) else → per-geometry thresholds.

    Returns
    -------
    all_edge_records : list[dict]
    threshold_info : dict
        {'mode': 'reference[...]'|'global'|'per-geometry', 'thresholds': {...}}
    """
    def compute_threshold_from_forces(forces):
        if not forces:
            return None
        return float(np.quantile(forces, quantile_threshold)) if quantile_threshold is not None else float(2 * np.mean(forces))

    def collect_forces(geom_graphs, graph_type):
        return [
            d['normal_force']
            for G in geom_graphs[graph_type]
            for _, _, d in G.edges(data=True)
            if 'normal_force' in d
        ]

    all_edge_records = []
    graph_type = 'core' if use_core_for_threshold else 'full'
    threshold_info = {"mode": None, "thresholds": {}}

    # --- Mode 1: Reference geometry threshold ---
    if reference_geometry_label is not None and reference_geometry_label in graph_dict:
        ref_forces = collect_forces(graph_dict[reference_geometry_label], graph_type)
        threshold = compute_threshold_from_forces(ref_forces)
        if threshold is not None:
            threshold_info["mode"] = f"reference[{reference_geometry_label}]"
            threshold_info["thresholds"][reference_geometry_label] = threshold
            print(f"🎯 Using threshold from {reference_geometry_label}: {threshold:.3f} "
                  f"({'quantile' if quantile_threshold is not None else '2×mean'})")

            for geom_graphs in graph_dict.values():
                for G_full, G_core in zip(geom_graphs['full'], geom_graphs['core']):
                    edge_records = _tag_high_force_edges_and_nodes(G_full, G_core, threshold)
                    all_edge_records.extend(edge_records)
            return all_edge_records, threshold_info
        else:
            print(f"⚠️ No forces in reference geometry '{reference_geometry_label}'. Falling back...")

    # --- Mode 2: Global threshold across all geometries ---
    if reference_geometry_label is None and use_global:
        all_forces = [
            d['normal_force']
            for geom_graphs in graph_dict.values()
            for G in geom_graphs[graph_type]
            for _, _, d in G.edges(data=True)
            if 'normal_force' in d
        ]
        threshold = compute_threshold_from_forces(all_forces)
        if threshold is None:
            print("⚠️ No forces found globally; nothing labeled.")
            return [], {"mode": "global", "thresholds": {}}

        threshold_info["mode"] = "global"
        threshold_info["thresholds"]["global"] = threshold
        print(f"🌐 Global high-force threshold: {threshold:.3f} "
              f"({'quantile' if quantile_threshold is not None else '2×mean'})")

        for geom_graphs in graph_dict.values():
            for G_full, G_core in zip(geom_graphs['full'], geom_graphs['core']):
                edge_records = _tag_high_force_edges_and_nodes(G_full, G_core, threshold)
                all_edge_records.extend(edge_records)

        return all_edge_records, threshold_info

    # --- Mode 3: Per-geometry thresholds ---
    threshold_info["mode"] = "per-geometry"
    for label, geom_graphs in graph_dict.items():
        forces = collect_forces(geom_graphs, graph_type)
        threshold = compute_threshold_from_forces(forces)
        if threshold is None:
            print(f"⚠️ No forces found for {label}; skipping.")
            continue
        threshold_info["thresholds"][label] = threshold
        print(f"📐 {label} high-force threshold: {threshold:.3f} "
              f"({'quantile' if quantile_threshold is not None else '2×mean'})")

        for G_full, G_core in zip(geom_graphs['full'], geom_graphs['core']):
            edge_records = _tag_high_force_edges_and_nodes(G_full, G_core, threshold)
            all_edge_records.extend(edge_records)

    return all_edge_records, threshold_info

# === Plot helpers ===
def _collect_forces_per_geometry(graph_dict, use_core_for_threshold):
    """Return {label: np.ndarray of normal forces} using core/full per the flag."""
    graph_type = 'core' if use_core_for_threshold else 'full'
    forces_by_label = {}
    for label, geom_graphs in graph_dict.items():
        forces = [
            d['normal_force']
            for G in geom_graphs[graph_type]
            for _, _, d in G.edges(data=True)
            if 'normal_force' in d
        ]
        forces_by_label[label] = np.asarray(forces, dtype=float) if forces else np.array([], dtype=float)
    return forces_by_label

def _threshold_per_label_from_info(threshold_info, labels):
    """Map each label to the threshold actually used for it, given threshold_info."""
    mode = threshold_info.get('mode')
    thr_map = {}
    if mode and mode.startswith('reference['):
        ref_label = mode[mode.find('[') + 1 : -1]
        thr = threshold_info['thresholds'].get(ref_label)
        for lab in labels:
            thr_map[lab] = thr
    elif mode == 'global':
        thr = threshold_info['thresholds'].get('global')
        for lab in labels:
            thr_map[lab] = thr
    else:
        # per-geometry
        for lab in labels:
            thr_map[lab] = threshold_info['thresholds'].get(lab)
    return thr_map, mode

def plot_force_distributions_row(
    graph_dict,
    threshold_info,
    use_core_for_threshold,
    out_dir,
    bins=100,
    logx=False,
    angle_order=None,
    column_layout=True,
    hist_linewidth=1.0,
    thr_linewidth=1.0,
    thr_color='red',
):
    """
    Save a figure with one connected (step) histogram per geometry, threshold as a dashed line.
    column_layout=True -> Nx1 column; False -> 1xN row.
    Enforces identical x/y limits across all panels.
    """
    forces_by_label = _collect_forces_per_geometry(graph_dict, use_core_for_threshold)
    labels = list(forces_by_label.keys())
    if angle_order:
        labels = [lab for lab in angle_order if lab in labels]
    thr_map, mode = _threshold_per_label_from_info(threshold_info, labels)

    # Shared x-limits & shared edges for consistency across panels
    all_forces = np.concatenate([forces_by_label[l] for l in labels if forces_by_label[l].size > 0]) \
                 if any(forces_by_label[l].size > 0 for l in labels) else np.array([], dtype=float)
    xmin = float(np.nanmin(all_forces)) if all_forces.size else 0.0
    xmax = float(np.nanmax(all_forces)) if all_forces.size else 1.0
    if xmax <= xmin:
        xmax = xmin + 1.0

    # Shared bin edges
    if logx:
        lo = max(xmin, np.nextafter(0, 1))  # avoid log(0)
        edges = np.logspace(np.log10(lo), np.log10(xmax), bins + 1)
    else:
        edges = np.linspace(xmin, xmax, bins + 1)

    # Precompute counts for ALL panels to get a global y-limit
    counts_dict = {}
    for lab in labels:
        data = forces_by_label[lab]
        if data.size:
            counts, _ = np.histogram(data, bins=edges)
        else:
            counts = np.zeros(len(edges) - 1, dtype=int)
        counts_dict[lab] = counts
    # Global y-limit (with a small headroom)
    y_max = max((c.max() for c in counts_dict.values()), default=1)
    if y_max <= 0:
        y_max = 1

    # Build figure with shared x and y
    if column_layout:
        fig, axes = plt.subplots(len(labels), 1, figsize=(5.0, 3.2 * len(labels)),
                                 sharex=True, sharey=True, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.2),
                                 sharex=True, sharey=True, constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]

    for idx, (ax, lab) in enumerate(zip(axes, labels)):
        counts = counts_dict[lab]
        if counts.sum() > 0:
            # Connected outline histogram (no fill, no seams)
            ax.stairs(counts, edges, fill=False, linewidth=hist_linewidth)
            thr = thr_map.get(lab)
            if thr is not None:
                ax.axvline(thr, linestyle='--', linewidth=thr_linewidth, color=thr_color)
                ax.text(0.98, 0.92, f"thr={thr:.3g}", transform=ax.transAxes, ha='right', va='top')
        else:
            ax.text(0.5, 0.5, 'No force data', transform=ax.transAxes, ha='center', va='center')

        ax.set_title(lab)
        if logx:
            ax.set_xscale('log')
        if column_layout:
            if idx == len(labels) - 1:
                ax.set_xlabel('Normal force')
            ax.set_ylabel('Count')
        else:
            ax.set_xlabel('Normal force')

        # Enforce identical limits
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, y_max * 1.05)

    if not column_layout:
        axes[0].set_ylabel('Count')

    fig.suptitle(f'Normal Force Distributions per Geometry (mode: {mode})', y=1.02)
    layout_tag = 'column' if column_layout else 'row'
    out_path = os.path.join(out_dir, f'normal_force_distributions_{(mode or "per-geometry").replace("/", "_")}_{layout_tag}.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved force distribution panel: {out_path}")
    return out_path


def _graph_id_fields(G):
    return {
        'geometry': G.graph.get('angle_label'),
        'sim_idx': G.graph.get('sim_idx'),
    }


PAIR_EDGE_FIELDNAMES = (
    'geometry',
    'sim_idx',
    'node1',
    'node2',
    'is_edge',
    'edge_connectivity',
    'edge_connectivity_with_walls',
)


def _save_pair_edge_connectivity_records(records, out_dir, geometry, sim_idx):
    pair_dir = os.path.join(out_dir, 'pair_edge_connectivity')
    os.makedirs(pair_dir, exist_ok=True)

    safe_geometry = str(geometry).replace('/', '_')
    filename = f'{safe_geometry}_sim_{int(sim_idx):03d}_pair_edge_connectivity.csv'
    csv_path = os.path.join(pair_dir, filename)

    num_pairs = 0
    with open(csv_path, 'w', newline='') as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=PAIR_EDGE_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
            num_pairs += 1

    return {
        'geometry': geometry,
        'sim_idx': sim_idx,
        'num_pairs': num_pairs,
        'pair_file': csv_path,
    }


def _build_feature_tables(graph_dict):
    node_records = []
    edge_records = []
    graph_records = []
    graph_array_records = []

    for graphs in graph_dict.values():
        for G_full, G_core in zip(graphs['full'], graphs['core']):
            graph_id = _graph_id_fields(G_full)

            graph_row = dict(graph_id)
            graph_array_row = dict(graph_id)
            for key, value in G_full.graph.items():
                if key in {'angle_label', 'sim_idx'}:
                    continue
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple, dict)):
                    graph_array_row[key] = value
                elif isinstance(value, np.generic):
                    graph_row[key] = value.item()
                else:
                    graph_row[key] = value
            graph_records.append(graph_row)
            if len(graph_array_row) > len(graph_id):
                graph_array_records.append(graph_array_row)

            for node, data in G_core.nodes(data=True):
                record = {
                    **graph_id,
                    'node_id': node,
                }
                for key, value in data.items():
                    if key == 'position' and _flatten_named_vector(record, key, value, ('x', 'y', 'z')):
                        continue
                    record[key] = _serialize_value(value)
                node_records.append(record)

            for u, v, data in G_full.edges(data=True):
                record = {
                    **graph_id,
                    'node1': u,
                    'node2': v,
                    'is_core_edge': bool(G_core.has_edge(u, v)),
                }
                for key, value in data.items():
                    if key == 'contact_location' and _flatten_named_vector(
                        record, key, value, ('contact_x', 'contact_y', 'contact_z')
                    ):
                        continue
                    if key == 'n_unit' and _flatten_named_vector(record, key, value, ('n_x', 'n_y', 'n_z')):
                        continue
                    if key == 't_unit' and _flatten_named_vector(record, key, value, ('t_x', 't_y', 't_z')):
                        continue
                    record[key] = _serialize_value(value)
                edge_records.append(record)

    return (
        pd.DataFrame(node_records),
        pd.DataFrame(edge_records),
        pd.DataFrame(graph_records),
        pd.DataFrame(graph_array_records),
    )


def _process_simulation_slice(
    label,
    sim_idx,
    force_data,
    pos_array,
    stress_array,
    roi_labels,
    start_idx_contact,
    num_contacts,
    start_idx_particle,
    num_particles,
    node_conn_n_jobs,
    node_conn_verbose,
    pair_edge_out_dir,
):
    print(f"[{label} sim {sim_idx:03d}] Starting slice with {num_particles} particles and {num_contacts} contacts...")
    G = nx.Graph()
    for pid in range(num_particles):
        pos = tuple(pos_array[pid])
        s11, s22, s33, s23, s13, s12 = stress_array[pid]
        hydro = (s11 + s22 + s33) / 3.0
        vm = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
        in_center = bool((roi_labels[pid] == 1).item() if np.ndim(roi_labels[pid]) else (roi_labels[pid] == 1))
        G.add_node(
            pid,
            position=pos,
            is_wall=False,
            in_center_region=in_center,
            stress_11=s11,
            stress_22=s22,
            stress_33=s33,
            stress_23=s23,
            stress_13=s13,
            stress_12=s12,
            stress_vm=vm,
            stress_hydro=hydro,
        )

    wall_node_counter = 0
    wall_pid_map = {}

    def get_node_id(pid, which_pid, contact_idx, wall_node_counter):
        if pid < 0:
            key = (sim_idx, contact_idx, which_pid)
            if key not in wall_pid_map:
                wall_id = f"{label}_wall_{sim_idx}_{wall_node_counter}"
                wall_pid_map[key] = wall_id
                wall_node_counter += 1
            else:
                wall_id = wall_pid_map[key]
            if wall_id not in G:
                G.add_node(wall_id)
            G.nodes[wall_id]['is_wall'] = True
            G.nodes[wall_id]['wall_label'] = pid
            return wall_id, wall_node_counter
        return pid, wall_node_counter

    for contact_idx, row in enumerate(force_data):
        pid1, pid2 = int(row[1]), int(row[2])
        node1, wall_node_counter = get_node_id(pid1, 'pid1', contact_idx, wall_node_counter)
        node2, wall_node_counter = get_node_id(pid2, 'pid2', contact_idx, wall_node_counter)
        if node1 in G and node2 in G:
            n_vec = (row[10], row[11], row[12])
            angle_deg = np.degrees(np.arccos(np.abs(np.clip(n_vec[2], -1.0, 1.0))))
            is_wall_contact = G.nodes[node1].get('is_wall', False) or G.nodes[node2].get('is_wall', False)
            G.add_edge(
                node1,
                node2,
                contact_location=(row[3], row[4], row[5]),
                delta=row[6],
                delta_t=row[7],
                normal_force=row[8],
                tangential_force=row[9],
                n_unit=n_vec,
                t_unit=(row[13], row[14], row[15]),
                angle_with_zz=angle_deg,
                is_wall_contact=is_wall_contact,
            )

    G.graph['sim_idx'] = sim_idx
    G.graph['angle_label'] = label
    G.graph['num_particles'] = num_particles
    G.graph['num_contacts'] = int(num_contacts)

    slice_entry = {
        'label': label,
        'sim_idx': sim_idx,
        'start_idx_contact': int(start_idx_contact),
        'num_contacts': int(num_contacts),
        'end_idx_contact': int(start_idx_contact + num_contacts),
        'start_idx_particle': int(start_idx_particle),
        'num_particles': int(num_particles),
        'end_idx_particle': int(start_idx_particle + num_particles),
    }

    G_core, pair_edge_index_entry = compute_dual_view_metrics(
        G,
        node_conn_n_jobs=node_conn_n_jobs,
        node_conn_verbose=node_conn_verbose,
        pair_edge_out_dir=pair_edge_out_dir,
    )

    print(f"[{label} sim {sim_idx:03d}] Finished slice.")

    return {
        'G_full': G,
        'G_core': G_core,
        'slice_entry': slice_entry,
        'pair_edge_index_entry': pair_edge_index_entry,
    }

def _folder_to_angle_label(folder_name):
    match = re.search(r'(\d+(?:\.\d+)?)', folder_name)
    if not match:
        return folder_name

    angle_value = float(match.group(1))
    if angle_value.is_integer():
        angle_text = str(int(angle_value))
    else:
        angle_text = match.group(1)
    return f'{angle_text}deg'


def _geometry_sort_key(folder_name):
    match = re.search(r'(\d+(?:\.\d+)?)', folder_name)
    angle_value = float(match.group(1)) if match else float('inf')
    return (angle_value, folder_name.lower())


def _discover_geometry_folders(data_root):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'Data directory not found: {data_root}')

    folder_names = [
        entry
        for entry in os.listdir(data_root)
        if not entry.startswith('.') and os.path.isdir(os.path.join(data_root, entry))
    ]
    folder_names.sort(key=_geometry_sort_key)
    return [(_folder_to_angle_label(folder_name), folder_name) for folder_name in folder_names]


def _find_mat_file(folder_path, preferred_names, required_tokens, optional=False):
    for name in preferred_names:
        candidate = os.path.join(folder_path, name)
        if os.path.exists(candidate):
            return candidate

    mat_files = [
        file_name for file_name in os.listdir(folder_path)
        if file_name.lower().endswith('.mat')
    ]
    for file_name in sorted(mat_files):
        lowered = file_name.lower()
        if all(token in lowered for token in required_tokens):
            return os.path.join(folder_path, file_name)

    if optional:
        return None

    raise FileNotFoundError(
        f'Could not find a .mat file in {folder_path} matching tokens {required_tokens}'
    )


def _load_mat_array(mat_file, preferred_keys, fallback_tokens, label, description):
    mat_data = scipy.io.loadmat(mat_file)
    key_used = next((key for key in preferred_keys if key in mat_data), None)

    if key_used is None:
        candidate_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        for key in candidate_keys:
            lowered = key.lower()
            if any(token in lowered for token in fallback_tokens):
                key_used = key
                break
        if key_used is None and len(candidate_keys) == 1:
            key_used = candidate_keys[0]

    if key_used is None:
        raise KeyError(f'Could not find {description} key in {mat_file}')

    print(
        f"Loaded {description} for {label}: file={os.path.basename(mat_file)}, key={key_used}"
    )
    return mat_data[key_used]


def _prepare_roi_array(roi_array, total_particles, label):
    if roi_array is None:
        print(f"ROI missing for {label}; using zeros for all {total_particles} particles.")
        return np.zeros(total_particles, dtype=int)

    roi_array = np.asarray(roi_array).reshape(-1)
    if roi_array.size == total_particles:
        return roi_array

    print(
        f"⚠️ ROI size mismatch for {label}: expected {total_particles}, got {roi_array.size}. "
        "Adjusting with zero padding/truncation."
    )
    adjusted = np.zeros(total_particles, dtype=roi_array.dtype if roi_array.size else int)
    adjusted[:min(total_particles, roi_array.size)] = roi_array[:min(total_particles, roi_array.size)]
    return adjusted


def _infer_particles_per_simulation(particle_positions, contact_counts, label):
    num_sims = len(contact_counts)
    total_particles = int(np.asarray(particle_positions).shape[0])

    if num_sims == 0:
        raise ValueError(f'No simulations found in contact counts for {label}.')
    if total_particles % num_sims != 0:
        raise ValueError(
            f'Cannot evenly split {total_particles} particles across {num_sims} simulations for {label}.'
        )

    return total_particles // num_sims


def main():
    graph_dict = {}
    slice_info = []
    pair_edge_index_records = []

    geometry_specs = _discover_geometry_folders(base_path)
    if not geometry_specs:
        raise FileNotFoundError(f'No geometry folders found under {base_path}')

    angle_labels = [label for label, _ in geometry_specs]
    print(f"Writing outputs to: {out_path}")
    print(f"Discovered geometries: {', '.join(angle_labels)}")

    for label, folder in geometry_specs:
        folder_path = os.path.join(base_path, folder)
        print(f"Processing {label} from {folder_path}...")

        force_file = _find_mat_file(folder_path, ('Forces.mat', 'forces.mat'), ('force',))
        count_file = _find_mat_file(folder_path, ('flengths.mat',), ('length',))
        pos_file = _find_mat_file(folder_path, ('Pos.mat', 'pos.mat'), ('pos',))
        stress_file = _find_mat_file(folder_path, ('AllStresses.mat',), ('stress',))
        roi_file = _find_mat_file(folder_path, ('ROI.mat', 'roi.mat'), ('roi',), optional=True)

        contact_forces = _load_mat_array(
            force_file,
            preferred_keys=('forces_collect',),
            fallback_tokens=('force',),
            label=label,
            description='force data',
        )
        contact_counts = np.asarray(
            _load_mat_array(
                count_file,
                preferred_keys=('f_lengths',),
                fallback_tokens=('length',),
                label=label,
                description='force length data',
            )
        ).reshape(-1)
        particle_positions = np.asarray(
            _load_mat_array(
                pos_file,
                preferred_keys=('Pos_collect',),
                fallback_tokens=('pos',),
                label=label,
                description='position data',
            )
        )
        stress_components = np.asarray(
            _load_mat_array(
                stress_file,
                preferred_keys=('sigma_collect',),
                fallback_tokens=('sigma', 'stress'),
                label=label,
                description='stress data',
            )
        )

        roi_array = None
        if roi_file is not None:
            try:
                roi_array = _load_mat_array(
                    roi_file,
                    preferred_keys=('ROI',),
                    fallback_tokens=('roi',),
                    label=label,
                    description='ROI data',
                )
            except Exception as exc:
                print(f"Error loading ROI for {label} from {roi_file}: {exc}")
                roi_array = None

        num_particles_per_sim = _infer_particles_per_simulation(
            particle_positions,
            contact_counts,
            label,
        )
        total_particles = num_particles_per_sim * len(contact_counts)

        if stress_components.shape[0] != total_particles:
            raise ValueError(
                f'Stress rows ({stress_components.shape[0]}) do not match total particles '
                f'({total_particles}) for {label}.'
            )

        roi_array = _prepare_roi_array(roi_array, total_particles, label)

        print(
            f"{label}: {len(contact_counts)} simulations, "
            f"{num_particles_per_sim} particles/simulation, "
            f"{int(np.sum(contact_counts))} contacts total."
        )

        graph_list_core, graph_list_full = [], []
        start_idx_contact = 0
        start_idx_particle = 0
        sim_tasks = []
        inner_node_conn_jobs = NODE_CONN_N_JOBS_WHEN_SIM_PARALLEL if RUN_SIM_PARALLEL else NODE_CONN_N_JOBS

        for sim_idx, num_contacts in enumerate(contact_counts):
            num_contacts = int(num_contacts)
            sim_tasks.append({
                'label': label,
                'sim_idx': sim_idx,
                'force_data': np.asarray(contact_forces[start_idx_contact:start_idx_contact + num_contacts]).copy(),
                'pos_array': np.asarray(
                    particle_positions[start_idx_particle:start_idx_particle + num_particles_per_sim]
                ).copy(),
                'stress_array': np.asarray(
                    stress_components[start_idx_particle:start_idx_particle + num_particles_per_sim]
                ).copy(),
                'roi_labels': np.asarray(
                    roi_array[start_idx_particle:start_idx_particle + num_particles_per_sim]
                ).copy(),
                'start_idx_contact': int(start_idx_contact),
                'num_contacts': num_contacts,
                'start_idx_particle': int(start_idx_particle),
                'num_particles': int(num_particles_per_sim),
                'node_conn_n_jobs': inner_node_conn_jobs,
                'node_conn_verbose': NODE_CONN_VERBOSE,
                'pair_edge_out_dir': out_path,
            })
            start_idx_contact += num_contacts
            start_idx_particle += num_particles_per_sim

        if RUN_SIM_PARALLEL and len(sim_tasks) > 1:
            print(f"Running {len(sim_tasks)} simulations for {label} in parallel with n_jobs={SIM_N_JOBS}...")
            sim_results = Parallel(n_jobs=SIM_N_JOBS, verbose=SIM_PARALLEL_VERBOSE)(
                delayed(_process_simulation_slice)(**task) for task in sim_tasks
            )
        else:
            sim_results = [_process_simulation_slice(**task) for task in sim_tasks]

        for task, result in zip(sim_tasks, sim_results):
            slice_info.append(result['slice_entry'])
            pair_edge_index_records.append(result['pair_edge_index_entry'])
            graph_list_core.append(result['G_core'])
            graph_list_full.append(result['G_full'])

        graph_dict[label] = {'core': graph_list_core, 'full': graph_list_full}
        print(f"→ {len(graph_list_core)} graphs built for {label}")

    high_force_edge_records, threshold_info = label_high_force_edges(
        graph_dict,
        use_global=USE_GLOBAL_HIGH_FORCE_THRESHOLD,
        use_core_for_threshold=USE_CORE_FOR_THRESHOLD,
        quantile_threshold=None,
        reference_geometry_label=REFERENCE_GEOMETRY_LABEL,
    )

    with open(os.path.join(out_path, 'graph_dict_labeled.pkl'), 'wb') as f:
        pickle.dump(graph_dict, f)
    print("✅ Saved labeled graph dictionary.")

    df_high_force_edges = pd.DataFrame(high_force_edge_records)
    df_high_force_edges.to_csv(os.path.join(out_path, 'high_force_edges.csv'), index=False)
    print("✅ Saved high-force edge list.")

    with open(os.path.join(out_path, "simulation_slices.txt"), "w") as f:
        header = ["label", "sim_idx", "start_idx_contact", "num_contacts", "end_idx_contact",
                  "start_idx_particle", "num_particles", "end_idx_particle"]
        f.write("\t".join(header) + "\n")
        for entry in slice_info:
            values = [str(entry[k]) for k in header]
            f.write("\t".join(values) + "\n")
    print("✅ Saved simulation slice metadata.")

    with open(os.path.join(out_path, 'high_force_threshold_info.pkl'), 'wb') as f:
        pickle.dump(threshold_info, f)
    print(f"✅ Threshold mode: {threshold_info['mode']}. Details: {threshold_info['thresholds']}")

    df_pair_edge_index = pd.DataFrame(pair_edge_index_records)
    df_pair_edge_index.to_csv(os.path.join(out_path, 'pair_edge_connectivity_index.csv'), index=False)
    print("✅ Saved pair edge-connectivity index.")

    df_nodes, df_edges, df_graphs, df_graph_arrays = _build_feature_tables(graph_dict)

    df_nodes.to_csv(os.path.join(out_path, 'node_features.csv'), index=False)
    print("✅ Saved node feature table.")

    df_edges.to_csv(os.path.join(out_path, 'edge_features.csv'), index=False)
    print("✅ Saved edge feature table.")

    df_graphs.to_csv(os.path.join(out_path, 'graph_features.csv'), index=False)
    print("✅ Saved graph feature table.")

    df_graph_arrays_serialized = df_graph_arrays.copy()
    for col in df_graph_arrays_serialized.columns:
        if col not in {'geometry', 'sim_idx'}:
            df_graph_arrays_serialized[col] = df_graph_arrays_serialized[col].map(_serialize_value)
    df_graph_arrays_serialized.to_csv(os.path.join(out_path, 'graph_feature_arrays.csv'), index=False)
    df_graph_arrays.to_pickle(os.path.join(out_path, 'graph_feature_arrays.pkl'))
    print("✅ Saved graph array table.")

    if PLOT_FORCE_DISTRIBUTIONS:
        plot_force_distributions_row(
            graph_dict,
            threshold_info,
            use_core_for_threshold=USE_CORE_FOR_THRESHOLD,
            out_dir=out_path,
            bins=FORCE_DIST_BINS,
            logx=FORCE_DIST_LOGX,
            angle_order=angle_labels,
            column_layout=FORCE_DIST_COLUMN,
            hist_linewidth=HIST_LINE_WIDTH,
            thr_linewidth=THRESHOLD_LINE_WIDTH,
            thr_color=THRESHOLD_LINE_COLOR,
        )


if __name__ == '__main__':
    main()
