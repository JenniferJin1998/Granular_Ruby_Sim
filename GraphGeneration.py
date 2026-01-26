import os
import numpy as np
import scipy.io
import networkx as nx
import pickle
import pandas as pd
from collections import defaultdict
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt

# === Config ===
base_path = '/Users/yfjin/Desktop/PhD/CHESS02142024/GraphNetwork/Simulation/sharing_data/T_new'
out_path = '/Users/yfjin/Desktop/PhD/CHESS02142024/GraphNetwork/AnalysisResults/T_new'
angle_labels = ['0deg', '15deg', '30deg', '45deg']
angle_folders = ['0degrees_all', '15degrees_all', '30degrees_all', '45degrees_all']
default_shape = (23200, 1)  # used when ROI.mat missing; will be flattened later
FIXED_NUM_PARTICLES = 464

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


# === Data containers ===
graph_dict = {}
slice_info = []

# === Helper to compute dual metrics ===
def compute_dual_view_metrics(G_full):
    G_core = G_full.copy()
    wall_nodes = [n for n, d in G_core.nodes(data=True) if d.get('is_wall', False)]
    G_core.remove_nodes_from(wall_nodes)

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

    # Assign dual-view node metrics (only to core nodes)
    for node in G_core.nodes():
        G_core.nodes[node]['degree'] = degree_core.get(node, 0)
        G_core.nodes[node]['closeness'] = closeness_core.get(node, 0.0)
        G_core.nodes[node]['betweenness'] = betweenness_core.get(node, 0.0)
        G_core.nodes[node]['clustering'] = clustering_core.get(node, 0.0)

        G_core.nodes[node]['degree_with_walls'] = degree_full.get(node, 0)
        G_core.nodes[node]['closeness_with_walls'] = closeness_full.get(node, 0.0)
        G_core.nodes[node]['betweenness_with_walls'] = betweenness_full.get(node, 0.0)
        G_core.nodes[node]['clustering_with_walls'] = clustering_full.get(node, 0.0)

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

    # Push metrics back onto the full graph for core nodes/edges
    for node in G_core.nodes:
        if node in G_full.nodes:
            G_full.nodes[node].update(G_core.nodes[node])
    for a, b in G_core.edges:
        if G_full.has_edge(a, b):
            G_full[a][b].update(G_core[a][b])

    return G_core

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

# === Main loop ===
for label, folder in zip(angle_labels, angle_folders):
    print(f"Processing {label}...")
    force_file = os.path.join(base_path, folder, 'forces.mat')
    count_file = os.path.join(base_path, folder, 'flengths.mat')
    pos_file = os.path.join(base_path, folder, 'pos.mat')
    stress_file = os.path.join(base_path, folder, 'AllStresses.mat')
    roi_file = os.path.join(base_path, folder, 'ROI.mat')

    # Load forces
    mat_data = scipy.io.loadmat(force_file)
    key_used = 'forces_collect' if 'forces_collect' in mat_data else None
    if not key_used:
        force_keys = [k for k in mat_data.keys() if 'force' in k.lower()]
        if force_keys:
            key_used = force_keys[0]
    contact_forces = mat_data[key_used] if key_used else None
    print(f"Loaded force data using key: {key_used}" if key_used else "⚠️ Force data key not found.")

    # Load contact counts
    mat_data = scipy.io.loadmat(count_file)
    key_used = 'f_lengths' if 'f_lengths' in mat_data else None
    if not key_used:
        force_keys = [k for k in mat_data.keys() if 'lengths' in k.lower()]
        if force_keys:
            key_used = force_keys[0]
    contact_counts = mat_data[key_used].squeeze() if key_used else None
    print(f"Loaded force length data using key: {key_used}" if key_used else "⚠️ Force length data key not found.")

    # Load positions
    mat_data = scipy.io.loadmat(pos_file)
    key_used = 'Pos_collect' if 'Pos_collect' in mat_data else None
    if not key_used:
        force_keys = [k for k in mat_data.keys() if 'pos' in k.lower()]
        if force_keys:
            key_used = force_keys[0]
    particle_positions = mat_data[key_used] if key_used else None
    print(f"Loaded position data using key: {key_used}" if key_used else "⚠️ Position data key not found.")

    # Load stresses
    mat_data = scipy.io.loadmat(stress_file)
    key_used = 'sigma_collect' if 'sigma_collect' in mat_data else None
    if not key_used:
        force_keys = [k for k in mat_data.keys() if 'sigma' in k.lower()]
        if force_keys:
            key_used = force_keys[0]
    stress_components = mat_data[key_used] if key_used else None
    print(f"Loaded stress data using key: {key_used}" if key_used else "⚠️ Stress data key not found.")

    # Load ROI (optional)
    if os.path.exists(roi_file):
        try:
            mat_data = scipy.io.loadmat(roi_file)
            key_used = 'ROI' if 'ROI' in mat_data else None
            if not key_used:
                force_keys = [k for k in mat_data.keys() if 'roi' in k.lower()]
                if force_keys:
                    key_used = force_keys[0]
            roi_array = mat_data[key_used] if key_used else None
            print(f"Loaded region separation data using key: {key_used}" if key_used else "⚠️ ROI key not found; using zeros.")
            if roi_array is None:
                roi_array = np.zeros(default_shape)
        except Exception as e:
            print(f"Error loading {roi_file}: {e}")
            roi_array = np.zeros(default_shape)
    else:
        print(f"ROI file not found: {roi_file}, setting to zeros.")
        roi_array = np.zeros(default_shape)

    # Ensure ROI is 1D for clean indexing
    roi_array = np.asarray(roi_array).reshape(-1)

    graph_list_core, graph_list_full = [], []
    start_idx_contact = start_idx_particle = 0

    for sim_idx, num_contacts in enumerate(contact_counts):
        force_data = contact_forces[start_idx_contact:start_idx_contact+num_contacts]
        pos_array = particle_positions[start_idx_particle:start_idx_particle+FIXED_NUM_PARTICLES]
        stress_array = stress_components[start_idx_particle:start_idx_particle+FIXED_NUM_PARTICLES]
        roi_labels = roi_array[start_idx_particle:start_idx_particle+FIXED_NUM_PARTICLES]

        G = nx.Graph()
        for pid in range(FIXED_NUM_PARTICLES):
            pos = tuple(pos_array[pid])
            s11, s22, s33, s23, s13, s12 = stress_array[pid]
            hydro = (s11 + s22 + s33) / 3.0
            vm = np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 3*(s12**2 + s13**2 + s23**2))
            in_center = bool((roi_labels[pid] == 1).item() if np.ndim(roi_labels[pid]) else (roi_labels[pid] == 1))
            G.add_node(
                pid,
                position=pos, is_wall=False, in_center_region=in_center,
                stress_11=s11, stress_22=s22, stress_33=s33,
                stress_23=s23, stress_13=s13, stress_12=s12,
                stress_vm=vm, stress_hydro=hydro
            )

        wall_node_counter = 0
        wall_pid_map = {}
        def get_node_id(pid, which_pid, wall_node_counter):
            # Negative pid → wall "node" placeholder
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
            node1, wall_node_counter = get_node_id(pid1, 'pid1', wall_node_counter)
            node2, wall_node_counter = get_node_id(pid2, 'pid2', wall_node_counter)
            if node1 in G and node2 in G:
                n_vec = (row[10], row[11], row[12])
                angle_deg = np.degrees(np.arccos(np.abs(np.clip(n_vec[2], -1.0, 1.0))))
                is_wall_contact = G.nodes[node1].get('is_wall', False) or G.nodes[node2].get('is_wall', False)
                G.add_edge(
                    node1, node2,
                    contact_location=(row[3], row[4], row[5]),
                    delta=row[6], delta_t=row[7],
                    normal_force=row[8], tangential_force=row[9],
                    n_unit=n_vec, t_unit=(row[13], row[14], row[15]),
                    angle_with_zz=angle_deg,
                    is_wall_contact=is_wall_contact
                )

        G.graph['sim_idx'] = sim_idx
        G.graph['angle_label'] = label
        G.graph['num_particles'] = FIXED_NUM_PARTICLES
        G.graph['num_contacts'] = int(num_contacts)

        slice_info.append({
            'label': label, 'sim_idx': sim_idx,
            'start_idx_contact': int(start_idx_contact),
            'num_contacts': int(num_contacts),
            'end_idx_contact': int(start_idx_contact + num_contacts),
            'start_idx_particle': int(start_idx_particle),
            'num_particles': FIXED_NUM_PARTICLES,
            'end_idx_particle': int(start_idx_particle + FIXED_NUM_PARTICLES)
        })

        G_core = compute_dual_view_metrics(G)
        graph_list_core.append(G_core)
        graph_list_full.append(G)
        start_idx_contact += int(num_contacts)
        start_idx_particle += FIXED_NUM_PARTICLES

    graph_dict[label] = {'core': graph_list_core, 'full': graph_list_full}
    print(f"→ {len(graph_list_core)} graphs built for {label}")

# === Label high-force edges (now with reference-geometry option) ===
high_force_edge_records, threshold_info = label_high_force_edges(
    graph_dict,
    use_global=USE_GLOBAL_HIGH_FORCE_THRESHOLD,
    use_core_for_threshold=USE_CORE_FOR_THRESHOLD,
    quantile_threshold=None,                 # set e.g. 0.95 to use percentile
    reference_geometry_label=REFERENCE_GEOMETRY_LABEL,
)

# === Save outputs ===
with open(os.path.join(out_path, 'graph_dict_labeled.pkl'), 'wb') as f:
    pickle.dump(graph_dict, f)
print("✅ Saved labeled graph dictionary.")

df_edges = pd.DataFrame(high_force_edge_records)
df_edges.to_csv(os.path.join(out_path, 'high_force_edges.csv'), index=False)
print("✅ Saved high-force edge list.")

with open(os.path.join(out_path, "simulation_slices.txt"), "w") as f:
    header = ["label", "sim_idx", "start_idx_contact", "num_contacts", "end_idx_contact",
              "start_idx_particle", "num_particles", "end_idx_particle"]
    f.write("\t".join(header) + "\n")
    for entry in slice_info:
        values = [str(entry[k]) for k in header]
        f.write("\t".join(values) + "\n")
print("✅ Saved simulation slice metadata.")

# Save threshold provenance for debugging / reproducibility
with open(os.path.join(out_path, 'high_force_threshold_info.pkl'), 'wb') as f:
    pickle.dump(threshold_info, f)
print(f"✅ Threshold mode: {threshold_info['mode']}. Details: {threshold_info['thresholds']}")

# === Plot distributions row with threshold(s) overlaid ===
if PLOT_FORCE_DISTRIBUTIONS:
    plot_force_distributions_row(
        graph_dict,
        threshold_info,
        use_core_for_threshold=USE_CORE_FOR_THRESHOLD,
        out_dir=out_path,
        bins=FORCE_DIST_BINS,
        logx=FORCE_DIST_LOGX,
        angle_order=angle_labels,
        column_layout=FORCE_DIST_COLUMN,             # ← column layout
        hist_linewidth=HIST_LINE_WIDTH,              # ← connected outline width
        thr_linewidth=THRESHOLD_LINE_WIDTH,          # ← thinner threshold line
        thr_color=THRESHOLD_LINE_COLOR,              # ← red threshold line
    )