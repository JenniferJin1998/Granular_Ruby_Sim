#!/usr/bin/env python3
"""Build and analyze full-network SC/BCC/FCC/HCP crystal references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PIPELINE_DIR = PROJECT_ROOT / "AnalysisScripts" / "Pipeline"
PERIODIC_PIPELINE_DIR = HERE / "PeriodicRubyPipeline"
DEFAULT_CONFIG = HERE / "crystal_reference_config.json"

# GraphPipelineCommon resolves several output settings while importing. Keep
# those incidental files inside the crystal-reference artifact tree.
os.environ.setdefault(
    "GRAPHGEN_OUT_PATH",
    str(PROJECT_ROOT / "AnalysisResults" / "crystal_references" / "0_graph_and_basic_stats" / "artifacts" / "pipeline"),
)
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PERIODIC_PIPELINE_DIR))

import GraphPipelineCommon as gp  # noqa: E402
import pipeline_common as pp  # noqa: E402
from crystal_common import STRUCTURES, crystal_graph, minimum_image_cell  # noqa: E402
from job4_bond_order import bond_order  # noqa: E402


NODE_PROPERTIES = [
    "degree",
    "closeness",
    "betweenness",
    "clustering",
    "avg_neighbor_degree",
    "principal_eigenvector",
    "fiedler",
    "avg_curvature_no_walls",
    "nfd",
    "nfd_r2",
    "q4",
    "q6",
    "q8",
    "q10",
    "q12",
    "qbar4",
    "qbar6",
    "w4",
    "what4",
    "w6",
    "what6",
    "mean_s6",
    "fraction_s6_above_threshold",
]
EDGE_PROPERTIES = [
    "angle_with_zz",
    "periodic_distance",
    "edge_connectivity",
    "node_connectivity",
    "curvature_no_walls",
    "s6",
]
GRAPH_PROPERTIES = [
    "num_nodes",
    "num_edges",
    "assortativity",
    "edge_connectivity_graph",
    "node_connectivity_graph",
    "num_components",
    "path_nodes",
    "avg_path",
    "diameter",
    "radius",
    "spectral_radius",
    "alg_connectivity",
    "fiedler_value",
    "loop_total",
    "loop_mean",
    "nfd_asymmetry",
    "nfd_tau_mean",
    "nfd_tau_std",
    "Q4",
    "Q6",
    "Q8",
    "Q10",
    "Q12",
    "mean_local_q4",
    "mean_local_q6",
    "mean_local_q8",
    "mean_local_q10",
    "mean_local_q12",
]
COLORS = {
    "0deg": "#1F77B4",
    "30deg": "#FF7F0E",
    "SC": "#2CA02C",
    "BCC": "#9467BD",
    "FCC": "#8C564B",
    "HCP": "#E377C2",
}
VIEWS = (
    (0, 0, "View along x"),
    (0, 90, "View along y"),
    (90, -90, "View along z"),
    (24, -52, "Perspective 3D"),
)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    root = Path(cfg.get("project_root") or PROJECT_ROOT).resolve()
    cfg["project_root"] = str(root)
    for key in (
        "output_root",
        "periodic_graph_pickle",
        "periodic_node_features",
        "periodic_edge_features",
        "periodic_graph_features",
        "periodic_bond_order_nodes",
        "periodic_bond_order_edges",
    ):
        cfg[key] = str((root / cfg[key]).resolve())
    return cfg


def fingerprint(cfg_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        cfg_path,
        PIPELINE_DIR / "GraphPipelineCommon.py",
        PERIODIC_PIPELINE_DIR / "crystal_common.py",
        PERIODIC_PIPELINE_DIR / "job4_bond_order.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def result_paths(cfg: dict) -> dict:
    root = Path(cfg["output_root"])
    basic = root / "0_graph_and_basic_stats"
    return {
        "root": root,
        "basic": basic,
        "raw": basic / "artifacts" / "raw_graphs",
        "patches": basic / "artifacts" / "patches",
        "pair": basic / "artifacts" / "pair_edge_connectivity",
        "graph_data": basic / "graph_data",
        "network": root / "1_network_property_comparison",
        "compare": root / "3_comparison_with_periodic",
    }


def manifest_valid(output: Path, digest: str) -> bool:
    manifest = Path(str(output) + ".complete.json")
    if not output.exists() or not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()).get("fingerprint") == digest
    except Exception:
        return False


def mark_complete(output: Path, digest: str, extra: dict | None = None) -> None:
    data = {"fingerprint": digest, "output": str(output), "size": output.stat().st_size}
    if extra:
        data.update(extra)
    atomic_json(Path(str(output) + ".complete.json"), data)


def boundary_distances(graph: nx.Graph, sources: set) -> dict:
    if not sources:
        return {node: np.nan for node in graph}
    return dict(nx.multi_source_dijkstra_path_length(graph, sources, weight=None))


def populate_geometric_edge_attributes(graph: nx.Graph, cell: np.ndarray, periodic_axes: list[int]) -> None:
    positions = {node: np.asarray(data["position"], float) for node, data in graph.nodes(data=True)}
    for u, v, data in graph.edges(data=True):
        raw = positions[v] - positions[u]
        delta = minimum_image_cell(raw[None, :], cell, tuple(periodic_axes))[0]
        distance = float(np.linalg.norm(delta))
        unit = delta / distance if distance else np.zeros(3)
        shift = delta - raw
        contact = positions[u] + 0.5 * delta
        data.update(
            contact_location=tuple(contact),
            n_unit=tuple(unit),
            angle_with_zz=float(np.degrees(np.arccos(np.clip(abs(unit[2]), 0.0, 1.0)))),
            is_wall_contact=False,
            periodic_dx=float(delta[0]),
            periodic_dy=float(delta[1]),
            periodic_dz=float(delta[2]),
            periodic_distance=distance,
            periodic_shift_x=float(shift[0]),
            periodic_shift_y=float(shift[1]),
            crosses_periodic_x=bool(abs(shift[0]) > 1e-15),
            crosses_periodic_y=bool(abs(shift[1]) > 1e-15),
            is_periodic_crossing=bool(abs(shift[0]) > 1e-15 or abs(shift[1]) > 1e-15),
        )


def add_top_bottom_walls(graph: nx.Graph, cfg: dict) -> tuple[set, set]:
    diameter = float(cfg["particle_diameter"])
    z_values = {node: float(data["position"][2]) for node, data in graph.nodes(data=True)}
    z_min, z_max = min(z_values.values()), max(z_values.values())
    tolerance = diameter * 1e-6
    bottom = {node for node, z in z_values.items() if abs(z - z_min) <= tolerance}
    top = {node for node, z in z_values.items() if abs(z - z_max) <= tolerance}
    for surface, nodes, direction, label in (
        ("bottom", bottom, -1.0, int(cfg["bottom_wall_label"])),
        ("top", top, 1.0, int(cfg["top_wall_label"])),
    ):
        for node in sorted(nodes):
            position = np.asarray(graph.nodes[node]["position"], float)
            wall_position = position.copy()
            wall_position[2] += direction * diameter / 2
            wall = f"{graph.graph['structure']}_{surface}_wall_{node}"
            graph.add_node(
                wall,
                position=tuple(wall_position),
                is_wall=True,
                wall_label=label,
                wall_surface=surface,
            )
            normal = np.array([0.0, 0.0, direction])
            graph.add_edge(
                node,
                wall,
                contact_location=tuple(wall_position),
                n_unit=tuple(normal),
                angle_with_zz=0.0,
                is_wall_contact=True,
                wall_label=label,
                wall_surface=surface,
                periodic_dx=0.0,
                periodic_dy=0.0,
                periodic_dz=float(direction * diameter / 2),
                periodic_distance=diameter / 2,
                periodic_shift_x=0.0,
                periodic_shift_y=0.0,
                crosses_periodic_x=False,
                crosses_periodic_y=False,
                is_periodic_crossing=False,
            )
    return bottom, top


def build_crystal(structure: str, cfg: dict, cfg_path: Path) -> Path:
    paths = result_paths(cfg)
    output = paths["raw"] / f"{structure}.pkl"
    digest = fingerprint(cfg_path)
    if manifest_valid(output, digest):
        print(f"Skipping valid crystal build: {structure}")
        return output
    graph = crystal_graph(
        structure,
        float(cfg["particle_diameter"]),
        float(cfg["contact_distance_tolerance_fraction"]),
    )
    cell = np.asarray(graph.graph["cell_matrix"], float)
    periodic_axes = [int(axis) for axis in cfg["periodic_axes"]]
    # Put every lateral coordinate in the displayed primary cell without
    # changing minimum-image contact vectors.
    for _, data in graph.nodes(data=True):
        position = np.asarray(data["position"], float)
        for axis in periodic_axes:
            position[axis] %= cell[axis, axis]
        data.update(position=tuple(position), is_wall=False, orientation=cfg["orientation"])
    graph.graph.update(
        structure=structure,
        angle_label=structure,
        orientation=cfg["orientation"],
        sim_idx=0,
        periodic_axes=periodic_axes,
        box_lengths={str(axis): float(cell[axis, axis]) for axis in range(3)},
        periodic_geometry_source="ideal_crystal_unit_cell",
        periodic_geometry_corrected=True,
        loading_axis=int(cfg["loading_axis"]),
        num_particles=graph.number_of_nodes(),
        has_force_data=False,
        has_stress_data=False,
    )
    populate_geometric_edge_attributes(graph, cell, periodic_axes)
    core_before_walls = graph.copy()
    bottom, top = add_top_bottom_walls(graph, cfg)
    for reference, sources in (("bottom", bottom), ("top", top), ("all", bottom | top)):
        distances = boundary_distances(core_before_walls, sources)
        for node in core_before_walls:
            graph.nodes[node][f"boundary_distance_{reference}"] = distances.get(node, np.nan)
    for node in core_before_walls:
        graph.nodes[node]["contacts_bottom_wall"] = node in bottom
        graph.nodes[node]["contacts_top_wall"] = node in top
    graph.graph.update(
        num_contacts=graph.number_of_edges(),
        bottom_surface_particles=len(bottom),
        top_surface_particles=len(top),
        bottom_wall_labels=[int(cfg["bottom_wall_label"])],
        top_wall_labels=[int(cfg["top_wall_label"])],
    )
    atomic_pickle(output, graph)
    mark_complete(
        output,
        digest,
        {
            "structure": structure,
            "particle_count": core_before_walls.number_of_nodes(),
            "particle_contact_count": core_before_walls.number_of_edges(),
            "bottom_surface_particles": len(bottom),
            "top_surface_particles": len(top),
        },
    )
    print(f"Built {structure}: {core_before_walls.number_of_nodes()} particles, {core_before_walls.number_of_edges()} particle contacts")
    return output


def compute_property(structure: str, group: str, cfg: dict, cfg_path: Path) -> Path:
    paths = result_paths(cfg)
    raw = paths["raw"] / f"{structure}.pkl"
    if not raw.exists():
        raise FileNotFoundError(f"Missing {raw}; run the build stage first")
    output = paths["patches"] / group / f"{structure}.pkl"
    digest = fingerprint(cfg_path)
    if manifest_valid(output, digest):
        print(f"Skipping valid {structure} {group} patch")
        return output
    graph = load_pickle(raw)
    cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if group == "topology":
        patch = gp.compute_topology_spectral_patch(graph)
    elif group == "loop":
        patch = gp.compute_loop_patch(graph)
    elif group == "pair_edge":
        pair_root = paths["pair"] / structure
        patch = gp.compute_pair_edge_patch(graph, str(pair_root), n_jobs=cpus, chunk_size=24)
    elif group == "node_connectivity":
        patch = gp.compute_node_connectivity_patch(graph, n_jobs=cpus, verbose=5)
    elif group == "curvature":
        patch = gp.compute_curvature_patch(graph)
    elif group == "nfd":
        patch = gp.compute_nfd_patch(graph)
    else:
        raise ValueError(f"Unknown property group: {group}")
    atomic_pickle(output, patch)
    extra = {"structure": structure, "group": group}
    if group == "pair_edge":
        extra.update(patch.get("extra", {}).get("pair_edge_index_entry", {}))
    mark_complete(output, digest, extra)
    print(f"Computed {structure} {group}")
    return output


def attach_bond_order(core: nx.Graph, full: nx.Graph, cfg: dict) -> None:
    cell = np.asarray(core.graph["cell_matrix"], float)
    periodic = tuple(int(axis) for axis in core.graph["periodic_axes"])
    nodes, edges, system = bond_order(core, cfg, cell, periodic)
    node_columns = [column for column in nodes if column not in {"node_id", "x", "y", "z"}]
    for row in nodes.to_dict("records"):
        node = row["node_id"]
        attrs = {column: row[column] for column in node_columns}
        core.nodes[node].update(attrs)
        full.nodes[node].update(attrs)
    for row in edges.to_dict("records"):
        u, v = row["source_node"], row["target_node"]
        if core.has_edge(u, v):
            core[u][v]["s6"] = row["s6"]
            full[u][v]["s6"] = row["s6"]
    core.graph.update(system)
    full.graph.update(system)


def layer_label(value, explicit: list[int]) -> str:
    if not np.isfinite(value):
        return "unreachable"
    integer = int(value)
    return str(integer) if integer in explicit else "rest"


def finalize_graphs(cfg: dict, cfg_path: Path) -> dict:
    paths = result_paths(cfg)
    digest = fingerprint(cfg_path)
    graph_dict = {}
    metadata = []
    boundary_rows = []
    pair_index = []
    for structure in cfg["structures"]:
        full = load_pickle(paths["raw"] / f"{structure}.pkl")
        core = gp._make_core_graph(full)
        for group in cfg["property_groups"]:
            patch_path = paths["patches"] / group / f"{structure}.pkl"
            if not manifest_valid(patch_path, digest):
                raise RuntimeError(f"Missing or stale property patch: {patch_path}")
            patch = load_pickle(patch_path)
            gp.apply_property_patch(full, core, patch)
            entry = patch.get("extra", {}).get("pair_edge_index_entry")
            if entry:
                pair_index.append(entry)
        for node in core:
            full.nodes[node].update(core.nodes[node])
        for u, v in core.edges():
            full[u][v].update(core[u][v])
        attach_bond_order(core, full, cfg)
        explicit = [int(value) for value in cfg["boundary_layers"]]
        for node, data in core.nodes(data=True):
            for reference in ("all", "top", "bottom"):
                data[f"boundary_layer_{reference}"] = layer_label(
                    float(data.get(f"boundary_distance_{reference}", np.nan)), explicit
                )
                full.nodes[node][f"boundary_layer_{reference}"] = data[f"boundary_layer_{reference}"]
            boundary_rows.append(
                {
                    "structure": structure,
                    "node_id": node,
                    "contacts_bottom_wall": bool(data.get("contacts_bottom_wall")),
                    "contacts_top_wall": bool(data.get("contacts_top_wall")),
                    **{f"boundary_distance_{ref}": data.get(f"boundary_distance_{ref}") for ref in ("all", "top", "bottom")},
                    **{f"boundary_layer_{ref}": data.get(f"boundary_layer_{ref}") for ref in ("all", "top", "bottom")},
                }
            )
        graph_dict[structure] = {"core": [core], "full": [full]}
        metadata.append(
            {
                "structure": structure,
                "orientation": cfg["orientation"],
                "particle_count": core.number_of_nodes(),
                "particle_contacts": core.number_of_edges(),
                "full_nodes": full.number_of_nodes(),
                "full_edges": full.number_of_edges(),
                "mean_coordination": 2 * core.number_of_edges() / core.number_of_nodes(),
                "packing_fraction": core.graph.get("packing_fraction"),
                "bottom_surface_particles": full.graph.get("bottom_surface_particles"),
                "top_surface_particles": full.graph.get("top_surface_particles"),
                "box_x": np.asarray(core.graph["cell_matrix"])[0, 0],
                "box_y": np.asarray(core.graph["cell_matrix"])[1, 1],
                "box_z": np.asarray(core.graph["cell_matrix"])[2, 2],
            }
        )
    data_root = paths["graph_data"]
    atomic_pickle(data_root / "graph_dict_labeled.pkl", graph_dict)
    node_frame, edge_frame, graph_frame, graph_arrays = gp._build_feature_tables(graph_dict)
    atomic_csv(data_root / "node_features.csv", node_frame)
    atomic_csv(data_root / "edge_features.csv", edge_frame)
    atomic_csv(data_root / "graph_features.csv", graph_frame)
    serial_arrays = graph_arrays.copy()
    for column in serial_arrays:
        if column not in {"geometry", "sim_idx"}:
            serial_arrays[column] = serial_arrays[column].map(gp._serialize_value)
    atomic_csv(data_root / "graph_feature_arrays.csv", serial_arrays)
    atomic_pickle(data_root / "graph_feature_arrays.pkl", graph_arrays)
    atomic_csv(data_root / "crystal_metadata.csv", pd.DataFrame(metadata))
    atomic_csv(data_root / "boundary_contacts_and_distances.csv", pd.DataFrame(boundary_rows))
    atomic_csv(data_root / "pair_edge_connectivity_index.csv", pd.DataFrame(pair_index))
    write_basic_readme(paths, cfg)
    analyze_crystal_properties(graph_dict, node_frame, edge_frame, graph_frame, cfg, paths)
    compare_with_periodic(node_frame, edge_frame, graph_frame, cfg, paths)
    return graph_dict


def finite(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return array[np.isfinite(array)]


def common_properties(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    return [prop for prop in requested if prop in frame and finite(frame[prop]).size]


def boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def summary_rows(frame: pd.DataFrame, group_column: str, properties: list[str], domain: str) -> list[dict]:
    rows = []
    for group, subset in frame.groupby(group_column, sort=False):
        for prop in properties:
            rows.append({"domain": domain, "group": group, "property": prop, **pp.distribution_summary(subset[prop])})
    return rows


def shared_edges(groups: list[np.ndarray], bins: int):
    return pp.shared_histogram_edges(groups, bins=bins)


def plot_group_distributions(
    groups: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    output: Path,
    bins: int,
) -> None:
    clean = {label: finite(values) for label, values in groups.items() if finite(values).size}
    if not clean:
        return
    edges = shared_edges(list(clean.values()), bins)
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    if edges is None:
        for label, values in clean.items():
            axis.axvline(float(np.mean(values)), color=COLORS.get(label), label=f"{label}: {np.mean(values):.4g}", linewidth=1.6)
        axis.set_ylabel("Constant-value reference")
    else:
        for label, values in clean.items():
            density, _ = np.histogram(values, bins=edges, density=True)
            axis.step(edges[:-1], density, where="post", color=COLORS.get(label), label=label, linewidth=1.5)
        axis.set_xlim(edges[0], edges[-1])
        axis.set_ylabel("Density")
    axis.set(title=title, xlabel=xlabel)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def split_periodic_segment(start: np.ndarray, delta: np.ndarray, lengths: dict[int, float]) -> list[np.ndarray]:
    current = start.copy()
    remaining = delta.copy()
    pieces = []
    for _ in range(3):
        endpoint = current + remaining
        crossings = []
        for axis in (0, 1):
            length = lengths[axis]
            if endpoint[axis] < 0 and remaining[axis] < 0:
                crossings.append(((0 - current[axis]) / remaining[axis], axis, length))
            elif endpoint[axis] > length and remaining[axis] > 0:
                crossings.append(((length - current[axis]) / remaining[axis], axis, 0.0))
        crossings = [item for item in crossings if 0 < item[0] < 1]
        if not crossings:
            pieces.append(np.vstack((current, endpoint)))
            break
        fraction = min(item[0] for item in crossings)
        hit = current + fraction * remaining
        pieces.append(np.vstack((current, hit)))
        remaining = (1 - fraction) * remaining
        current = hit.copy()
        for other_fraction, axis, wrapped in crossings:
            if np.isclose(other_fraction, fraction):
                current[axis] = wrapped
    return pieces


def plot_crystal_four_views(graph: nx.Graph, structure: str, output: Path) -> None:
    nodes = list(graph)
    xyz = np.asarray([graph.nodes[node]["position"] for node in nodes], float)
    index = {node: idx for idx, node in enumerate(nodes)}
    cell = np.asarray(graph.graph["cell_matrix"], float)
    lengths = {axis: float(cell[axis, axis]) for axis in range(3)}
    segments = []
    for u, v in graph.edges():
        start = xyz[index[u]]
        delta = minimum_image_cell((xyz[index[v]] - start)[None, :], cell, (0, 1))[0]
        segments.extend(split_periodic_segment(start, delta, lengths))
    mins = np.array([0.0, 0.0, xyz[:, 2].min()])
    maxs = np.array([lengths[0], lengths[1], xyz[:, 2].max()])
    span = np.maximum(maxs - mins, np.finfo(float).eps)
    fig = plt.figure(figsize=(12, 10))
    axes = [fig.add_subplot(2, 2, idx + 1, projection="3d") for idx in range(4)]
    for axis, (elev, azim, view_title) in zip(axes, VIEWS):
        axis.add_collection3d(Line3DCollection(segments, colors="#4C78A8", linewidths=0.35, alpha=0.42, rasterized=True))
        axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=3, color="#D62728", alpha=0.82, depthshade=False, rasterized=True)
        axis.view_init(elev=elev, azim=azim)
        axis.set_proj_type("persp" if view_title == "Perspective 3D" else "ortho")
        axis.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]), zlim=(mins[2], maxs[2]), title=view_title)
        axis.set_box_aspect(span)
        axis.set_axis_off()
    fig.suptitle(f"{structure} crystal reference: 0° orientation, periodic x/y")
    fig.subplots_adjust(bottom=0.04, top=0.92, wspace=0.02, hspace=0.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_crystal_properties(
    graph_dict: dict,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    graphs: pd.DataFrame,
    cfg: dict,
    paths: dict,
) -> None:
    root = paths["network"]
    core_edges = edges.loc[boolean_mask(edges["is_core_edge"])].copy()
    node_props = common_properties(nodes, NODE_PROPERTIES)
    edge_props = common_properties(core_edges, EDGE_PROPERTIES)
    graph_props = common_properties(graphs, GRAPH_PROPERTIES)
    summaries = []
    summaries.extend(summary_rows(nodes, "geometry", node_props, "node"))
    summaries.extend(summary_rows(core_edges, "geometry", edge_props, "edge"))
    summaries.extend(summary_rows(graphs, "geometry", graph_props, "graph"))
    atomic_csv(root / "tables" / "property_distribution_summaries.csv", pd.DataFrame(summaries))
    bins = int(cfg["histogram_bins"])
    for domain, frame, properties in (
        ("node", nodes, node_props),
        ("edge", core_edges, edge_props),
        ("graph", graphs, graph_props),
    ):
        for prop in properties:
            groups = {structure: frame.loc[frame.geometry == structure, prop] for structure in cfg["structures"]}
            plot_group_distributions(
                groups,
                f"Crystal-reference {domain} property: {prop.replace('_', ' ')}",
                prop.replace("_", " "),
                root / "distributions" / domain / f"{safe_name(prop)}.png",
                bins,
            )
    boundary_summaries = []
    layer_order = [str(value) for value in cfg["boundary_layers"]] + ["rest"]
    for reference in ("all", "top", "bottom"):
        layer_column = f"boundary_layer_{reference}"
        for structure in cfg["structures"]:
            structure_frame = nodes[nodes.geometry == structure]
            for layer in layer_order:
                layer_frame = structure_frame[structure_frame[layer_column].astype(str) == layer]
                for prop in node_props:
                    boundary_summaries.append(
                        {
                            "reference": reference,
                            "structure": structure,
                            "layer": layer,
                            "property": prop,
                            **pp.distribution_summary(layer_frame[prop]),
                        }
                    )
        for prop in node_props:
            all_values = [
                nodes.loc[(nodes.geometry == structure) & (nodes[layer_column].astype(str) == layer), prop]
                for structure in cfg["structures"]
                for layer in layer_order
            ]
            edges_shared = shared_edges([finite(values) for values in all_values], bins)
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
            for axis, structure in zip(axes.flat, cfg["structures"]):
                subset = nodes[nodes.geometry == structure]
                for idx, layer in enumerate(layer_order):
                    values = finite(subset.loc[subset[layer_column].astype(str) == layer, prop])
                    if not len(values):
                        continue
                    color = plt.get_cmap("viridis")(idx / max(1, len(layer_order) - 1))
                    if edges_shared is None:
                        axis.axvline(np.mean(values), color=color, label=layer)
                    else:
                        density, _ = np.histogram(values, bins=edges_shared, density=True)
                        axis.step(edges_shared[:-1], density, where="post", color=color, label=layer)
                axis.set_title(structure)
                axis.grid(alpha=0.18)
            axes[0, 0].legend(title="Graph distance", frameon=False)
            fig.suptitle(f"{prop.replace('_', ' ')} by {reference} boundary distance")
            fig.supxlabel(prop.replace("_", " "))
            fig.supylabel("Density")
            fig.tight_layout()
            output = root / "boundary_layers" / reference / "distributions" / f"{safe_name(prop)}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=210)
            plt.close(fig)
    atomic_csv(root / "boundary_layers" / "tables" / "boundary_layer_summaries.csv", pd.DataFrame(boundary_summaries))
    for structure in cfg["structures"]:
        plot_crystal_four_views(
            graph_dict[structure]["core"][0],
            structure,
            root / "sample_systems" / f"{structure}_four_views.png",
        )
    write_network_readme(paths, cfg, node_props, edge_props, graph_props)


def parse_bond_system(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["system"].str.extract(r"ruby_(?P<geometry>.+)_sim(?P<sim_idx>\d+)")
    result = frame.loc[parsed.geometry.notna()].copy()
    result["geometry"] = parsed.loc[parsed.geometry.notna(), "geometry"].to_numpy()
    result["sim_idx"] = parsed.loc[parsed.geometry.notna(), "sim_idx"].astype(int).to_numpy()
    return result


def add_periodic_bond_order(nodes: pd.DataFrame, edges: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bond_nodes = parse_bond_system(pd.read_csv(cfg["periodic_bond_order_nodes"]))
    bond_node_columns = [column for column in NODE_PROPERTIES if column in bond_nodes]
    nodes = nodes.merge(
        bond_nodes[["geometry", "sim_idx", "node_id", *bond_node_columns]],
        on=["geometry", "sim_idx", "node_id"],
        how="left",
        suffixes=("", "_bond"),
    )
    bond_edges = parse_bond_system(pd.read_csv(cfg["periodic_bond_order_edges"]))
    if "s6" in bond_edges:
        bond_source = pd.to_numeric(bond_edges["source_node"], errors="coerce")
        bond_target = pd.to_numeric(bond_edges["target_node"], errors="coerce")
        edge_source = pd.to_numeric(edges["node1"], errors="coerce")
        edge_target = pd.to_numeric(edges["node2"], errors="coerce")
        bond_edges["edge_low"] = np.minimum(bond_source, bond_target)
        bond_edges["edge_high"] = np.maximum(bond_source, bond_target)
        edges["edge_low"] = np.minimum(edge_source, edge_target)
        edges["edge_high"] = np.maximum(edge_source, edge_target)
        edges = edges.merge(
            bond_edges[["geometry", "sim_idx", "edge_low", "edge_high", "s6"]],
            on=["geometry", "sim_idx", "edge_low", "edge_high"],
            how="left",
        )
    return nodes, edges


def plot_reference_lines(
    ruby_groups: dict[str, np.ndarray],
    crystal_groups: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    output: Path,
    bins: int,
) -> None:
    all_groups = [finite(values) for values in [*ruby_groups.values(), *crystal_groups.values()]]
    edges = shared_edges(all_groups, bins)
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    if edges is None:
        for label, values in ruby_groups.items():
            values = finite(values)
            if len(values):
                axis.axvline(np.mean(values), color=COLORS[label], linewidth=2, label=label)
    else:
        for label, values in ruby_groups.items():
            values = finite(values)
            if not len(values):
                continue
            density, _ = np.histogram(values, bins=edges, density=True)
            axis.step(edges[:-1], density, where="post", color=COLORS[label], linewidth=1.8, label=label)
        axis.set_xlim(edges[0], edges[-1])
    for label, values in crystal_groups.items():
        values = finite(values)
        if len(values):
            axis.axvline(np.mean(values), color=COLORS[label], linestyle="--", linewidth=1.35, label=f"{label} mean")
    axis.set(title=title, xlabel=xlabel, ylabel="Ruby pooled density")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_simulation_means(frame: pd.DataFrame, prop: str, output: Path, title: str, group_order: list[str]) -> None:
    groups = [finite(frame.loc[frame.comparison_group == group, prop]) for group in group_order]
    if not any(len(values) for values in groups):
        return
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.boxplot(groups, labels=group_order, showmeans=True)
    rng = np.random.default_rng(20260921)
    for idx, (label, values) in enumerate(zip(group_order, groups), start=1):
        axis.scatter(idx + rng.uniform(-0.055, 0.055, len(values)), values, color=COLORS[label], s=20, alpha=0.72)
    axis.set(title=title, ylabel=prop.replace("_", " "))
    axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def compare_domain(
    ruby: pd.DataFrame,
    crystal: pd.DataFrame,
    properties: list[str],
    domain: str,
    cfg: dict,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ruby = ruby.copy()
    crystal = crystal.copy()
    ruby["comparison_group"] = ruby["geometry"]
    crystal["comparison_group"] = crystal["geometry"]
    ruby["sample_id"] = ruby.geometry.astype(str) + "_sim" + ruby.sim_idx.astype(int).astype(str)
    crystal["sample_id"] = crystal.geometry.astype(str)
    pooled = pd.concat([ruby, crystal], ignore_index=True, sort=False)
    group_order = [*cfg["comparison_angles"], *cfg["structures"]]
    summaries = []
    mean_summaries = []
    bins = int(cfg["histogram_bins"])
    for prop in properties:
        groups = {group: pooled.loc[pooled.comparison_group == group, prop] for group in group_order}
        plot_group_distributions(
            groups,
            f"Periodic Ruby and crystal references: {prop.replace('_', ' ')}",
            prop.replace("_", " "),
            root / "distributions" / domain / f"{safe_name(prop)}.png",
            bins,
        )
        plot_reference_lines(
            {angle: ruby.loc[ruby.geometry == angle, prop] for angle in cfg["comparison_angles"]},
            {structure: crystal.loc[crystal.geometry == structure, prop] for structure in cfg["structures"]},
            f"Periodic Ruby distributions with crystal means: {prop.replace('_', ' ')}",
            prop.replace("_", " "),
            root / "ruby_distributions_with_crystal_reference_lines" / domain / f"{safe_name(prop)}.png",
            bins,
        )
        sample_means = pooled.groupby(["comparison_group", "sample_id"], sort=False)[prop].mean().reset_index()
        plot_simulation_means(
            sample_means,
            prop,
            root / "simulation_mean_boxplots" / domain / f"{safe_name(prop)}.png",
            f"Per-system mean: {prop.replace('_', ' ')}",
            group_order,
        )
        for group in group_order:
            summaries.append(
                {"domain": domain, "group": group, "property": prop, **pp.distribution_summary(groups[group])}
            )
            mean_summaries.append(
                {
                    "domain": domain,
                    "group": group,
                    "property": prop,
                    **pp.distribution_summary(sample_means.loc[sample_means.comparison_group == group, prop]),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(mean_summaries)


def compare_with_periodic(
    crystal_nodes: pd.DataFrame,
    crystal_edges: pd.DataFrame,
    crystal_graphs: pd.DataFrame,
    cfg: dict,
    paths: dict,
) -> None:
    root = paths["compare"]
    ruby_nodes = pd.read_csv(cfg["periodic_node_features"])
    ruby_edges = pd.read_csv(cfg["periodic_edge_features"])
    ruby_graphs = pd.read_csv(cfg["periodic_graph_features"])
    ruby_nodes = ruby_nodes[ruby_nodes.geometry.isin(cfg["comparison_angles"])].copy()
    ruby_edges = ruby_edges[ruby_edges.geometry.isin(cfg["comparison_angles"])].copy()
    ruby_graphs = ruby_graphs[ruby_graphs.geometry.isin(cfg["comparison_angles"])].copy()
    ruby_edges = ruby_edges[boolean_mask(ruby_edges["is_core_edge"])].copy()
    ruby_nodes, ruby_edges = add_periodic_bond_order(ruby_nodes, ruby_edges, cfg)
    crystal_edges = crystal_edges[boolean_mask(crystal_edges["is_core_edge"])].copy()
    node_props = [prop for prop in NODE_PROPERTIES if prop in ruby_nodes and prop in crystal_nodes and finite(ruby_nodes[prop]).size and finite(crystal_nodes[prop]).size]
    edge_props = [prop for prop in EDGE_PROPERTIES if prop in ruby_edges and prop in crystal_edges and finite(ruby_edges[prop]).size and finite(crystal_edges[prop]).size]
    graph_props = [prop for prop in GRAPH_PROPERTIES if prop in ruby_graphs and prop in crystal_graphs and finite(ruby_graphs[prop]).size and finite(crystal_graphs[prop]).size]
    summaries = []
    mean_summaries = []
    for ruby, crystal, properties, domain in (
        (ruby_nodes, crystal_nodes, node_props, "node"),
        (ruby_edges, crystal_edges, edge_props, "edge"),
        (ruby_graphs, crystal_graphs, graph_props, "graph"),
    ):
        pooled, means = compare_domain(ruby, crystal, properties, domain, cfg, root)
        summaries.append(pooled)
        mean_summaries.append(means)
    atomic_csv(root / "tables" / "pooled_distribution_summaries.csv", pd.concat(summaries, ignore_index=True))
    atomic_csv(root / "tables" / "per_system_mean_summaries.csv", pd.concat(mean_summaries, ignore_index=True))
    inventory = pd.DataFrame(
        [
            {"domain": "node", "property": prop} for prop in node_props
        ]
        + [{"domain": "edge", "property": prop} for prop in edge_props]
        + [{"domain": "graph", "property": prop} for prop in graph_props]
    )
    atomic_csv(root / "tables" / "compared_property_inventory.csv", inventory)
    write_comparison_readme(paths, cfg, node_props, edge_props, graph_props)


def write_basic_readme(paths: dict, cfg: dict) -> None:
    text = f"""# 0 — Crystal graph construction and basic statistics

SC, BCC, FCC, and HCP ideal crystals are generated at 0° orientation with the
same particle diameter as the Ruby simulations (`{cfg['particle_diameter']} m`).
The systems contain approximately the same number of particles as a periodic
Ruby sample. Contacts are defined by minimum-image distance within
±{cfg['contact_distance_tolerance_fraction']:.0%} of one particle diameter.

The x and y directions are periodic. There are no lateral wall nodes or lateral
boundary edges. Particles in the lowest and highest z planes are connected to
synthetic bottom and top wall-contact nodes in the full graph. The canonical
`core` graph contains particles and particle-particle contacts only.

`graph_data/` contains the labeled graph dictionary, node/edge/graph feature
tables, array-valued graph features, construction metadata, boundary contacts,
and the all-pairs edge-connectivity index. `artifacts/` contains restartable raw
graphs, property patches, and all-pairs connectivity tables.

The topology, spectral, loop, edge/node connectivity, Ollivier-Ricci curvature,
NFD, and bond-order calculations use the same implementations as the periodic
Ruby graph pipeline. Force and stress attributes are intentionally absent
because an ideal geometric reference does not define them.
"""
    paths["basic"].mkdir(parents=True, exist_ok=True)
    (paths["basic"] / "README.md").write_text(text)


def write_network_readme(paths: dict, cfg: dict, node_props: list[str], edge_props: list[str], graph_props: list[str]) -> None:
    text = f"""# 1 — Crystal network-property comparison

This section compares whole-graph properties among SC, BCC, FCC, and HCP.
It contains pooled distributions for {len(node_props)} node properties,
{len(edge_props)} edge properties, and {len(graph_props)} graph properties.
Every distribution comparison uses one shared bin grid for all four crystals.

`boundary_layers/` repeats node-property distributions using graph distance
from combined top+bottom, top-only, and bottom-only surface particles. Distance
0 means direct wall contact; distances 1 and 2 are one and two graph steps away;
all larger distances are grouped as `rest`.

`sample_systems/` contains four-view graph renderings. Periodic x/y contacts
are drawn with minimum-image segments and no artificial system-length bonds.
There is no high/non-high-force or force-cluster analysis for these references.
"""
    paths["network"].mkdir(parents=True, exist_ok=True)
    (paths["network"] / "README.md").write_text(text)


def write_comparison_readme(paths: dict, cfg: dict, node_props: list[str], edge_props: list[str], graph_props: list[str]) -> None:
    text = f"""# 3 — Crystal references versus periodic Ruby

This section compares the four ideal crystals with the 20 periodic Ruby 0°
systems and 20 periodic Ruby 30° systems. It includes {len(node_props)} common
node properties, {len(edge_props)} common edge properties, and
{len(graph_props)} common graph properties.

- `distributions/` overlays Ruby and crystal distributions using the same bins.
- `ruby_distributions_with_crystal_reference_lines/` shows the Ruby 0°/30°
  distributions with dashed lines at each crystal mean.
- `simulation_mean_boxplots/` treats each Ruby simulation as one replicate and
  shows each deterministic crystal as a reference point, not as a replicate
  population.
- `tables/` stores pooled and per-system summaries and the property inventory.

No inferential p-value is assigned to Ruby-versus-crystal comparisons because
each ideal lattice is one deterministic reference system.
"""
    paths["compare"].mkdir(parents=True, exist_ok=True)
    (paths["compare"] / "README.md").write_text(text)
    root_text = """# Crystal references

The numbered layout mirrors the physical-state datasets:

```text
0_graph_and_basic_stats/
1_network_property_comparison/
3_comparison_with_periodic/
```

Section 2 is intentionally absent because ideal crystals have no force data.
See each numbered folder's README for methods and output organization.
"""
    (paths["root"] / "README.md").write_text(root_text)


def choose_structure(cfg: dict, explicit: str | None) -> str:
    if explicit:
        if explicit not in cfg["structures"]:
            raise ValueError(f"Unknown structure {explicit}; choose from {cfg['structures']}")
        return explicit
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        raise ValueError("Provide --structure or SLURM_ARRAY_TASK_ID")
    return cfg["structures"][int(task)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("build", "property", "finalize", "all"), required=True)
    parser.add_argument("--structure", choices=STRUCTURES)
    parser.add_argument("--group", choices=("topology", "loop", "pair_edge", "node_connectivity", "curvature", "nfd"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "structure": args.structure,
                    "group": args.group,
                    "structures": cfg["structures"],
                    "property_groups": cfg["property_groups"],
                    "output_root": cfg["output_root"],
                },
                indent=2,
            )
        )
        return
    if args.stage == "build":
        targets = [args.structure] if args.structure else cfg["structures"]
        for structure in targets:
            build_crystal(structure, cfg, cfg_path)
    elif args.stage == "property":
        if not args.group:
            raise SystemExit("--group is required for the property stage")
        compute_property(choose_structure(cfg, args.structure), args.group, cfg, cfg_path)
    elif args.stage == "finalize":
        finalize_graphs(cfg, cfg_path)
    else:
        for structure in cfg["structures"]:
            build_crystal(structure, cfg, cfg_path)
            for group in cfg["property_groups"]:
                compute_property(structure, group, cfg, cfg_path)
        finalize_graphs(cfg, cfg_path)


if __name__ == "__main__":
    main()
