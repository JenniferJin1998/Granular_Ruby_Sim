#!/usr/bin/env python3
"""Paper-inspired force-threshold percolation sweeps for 3D contact graphs."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import pickle
import tempfile
from collections import deque
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "force_threshold_percolation_config.json"
VIEWS = (
    (0, 0, "View along x"),
    (0, 90, "View along y"),
    (90, -90, "View along z"),
    (24, -52, "Perspective 3D"),
)
CURVE_PROPERTIES = {
    "cluster_count": "Number of clusters",
    "largest_cluster_size": "Largest cluster size",
    "second_to_largest_ratio": "Second/largest size ratio",
    "largest_cluster_fraction": "Largest cluster / all particles",
    "relative_length_x": "Largest-cluster relative length: x",
    "relative_length_y": "Largest-cluster relative length: y",
    "relative_length_z": "Largest-cluster relative length: z",
    "mean_degree": "Mean degree of strong network",
    "clustering_coefficient": "Mean clustering coefficient",
    "largest_diameter_normalized": r"Largest-cluster diameter / $N_p^{1/3}$",
    "largest_radius_normalized": r"Largest-cluster radius / $N_p^{1/3}$",
    "largest_mean_shortest_path": "Largest-cluster mean shortest path",
    "betweenness_ratio": r"Mean betweenness ratio $B_n/B_0$",
    "largest_spans_loading": "Largest-cluster spanning probability",
}


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    project = Path(cfg.get("project_root") or PROJECT_ROOT).resolve()
    cfg["project_root"] = str(project)
    for dataset in cfg["datasets"].values():
        dataset["graph_pickle"] = str((project / dataset["graph_pickle"]).resolve())
        dataset["output_root"] = str((project / dataset["output_root"]).resolve())
        dataset["box_lengths"] = {int(k): float(v) for k, v in dataset.get("box_lengths", {}).items()}
        dataset["periodic_axes"] = [int(v) for v in dataset.get("periodic_axes", [])]
    cfg["combined_output_root"] = str((project / cfg["combined_output_root"]).resolve())
    return cfg


def fingerprint(cfg: dict, config_path: Path) -> str:
    payload = config_path.read_bytes() + Path(__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def threshold_values(cfg: dict, override: str | None = None) -> list[float]:
    if override:
        return [float(value) for value in override.split(",")]
    sweep = cfg["sweep"]
    decimals = int(sweep["decimals"])
    count = int(round((sweep["stop"] - sweep["start"]) / sweep["step"])) + 1
    return [round(sweep["start"] + i * sweep["step"], decimals) for i in range(count)]


def n_folder(value: float, decimals: int) -> str:
    return f"n_{value:.{decimals}f}".replace("-", "m").replace(".", "p")


def all_tasks(cfg: dict) -> list[tuple[str, str, int]]:
    tasks = []
    for dataset_key, dataset in cfg["datasets"].items():
        for angle in dataset["angles"]:
            for sim_idx in range(int(dataset["simulations_per_angle"])):
                tasks.append((dataset_key, angle, sim_idx))
    return tasks


def load_graphs(path: str) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def is_wall_node(node, data: dict) -> bool:
    return bool(data.get("is_wall", False)) or (
        isinstance(node, (int, np.integer)) and int(node) < 0
    )


def wall_contact_particles(full: nx.Graph, labels: list[int]) -> set:
    labels = {int(v) for v in labels}
    particles = set()
    for node, data in full.nodes(data=True):
        if not is_wall_node(node, data):
            continue
        label = int(data.get("wall_label", node))
        if label not in labels:
            continue
        particles.update(neighbor for neighbor in full.neighbors(node) if not is_wall_node(neighbor, full.nodes[neighbor]))
    return particles


def minimum_image(delta: np.ndarray, box_lengths: dict[int, float], periodic_axes: list[int]) -> np.ndarray:
    result = np.asarray(delta, dtype=float).copy()
    for axis in periodic_axes:
        length = box_lengths[axis]
        result[axis] -= length * np.round(result[axis] / length)
    return result


def unwrap_component(
    graph: nx.Graph,
    box_lengths: dict[int, float],
    periodic_axes: list[int],
) -> tuple[dict, dict[int, bool]]:
    if not graph:
        return {}, {axis: False for axis in periodic_axes}
    raw = {node: np.asarray(graph.nodes[node]["position"], dtype=float) for node in graph}
    root = next(iter(graph))
    unwrapped = {root: raw[root].copy()}
    wraps = {axis: False for axis in periodic_axes}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            candidate = unwrapped[node] + minimum_image(
                raw[neighbor] - raw[node], box_lengths, periodic_axes
            )
            if neighbor not in unwrapped:
                unwrapped[neighbor] = candidate
                queue.append(neighbor)
                continue
            mismatch = candidate - unwrapped[neighbor]
            for axis in periodic_axes:
                if abs(mismatch[axis]) > 0.5 * box_lengths[axis]:
                    wraps[axis] = True
    return unwrapped, wraps


def component_lengths(
    graph: nx.Graph,
    all_spans: np.ndarray,
    box_lengths: dict[int, float],
    periodic_axes: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[int, bool]]:
    if not graph:
        return np.zeros(3), np.zeros(3), {axis: False for axis in periodic_axes}
    unwrapped, wraps = unwrap_component(graph, box_lengths, periodic_axes)
    xyz = np.asarray([unwrapped[node] for node in graph], dtype=float)
    extents = np.ptp(xyz, axis=0)
    relative = np.zeros(3, dtype=float)
    for axis in range(3):
        denominator = box_lengths.get(axis, all_spans[axis]) if axis in periodic_axes else all_spans[axis]
        if axis in periodic_axes and wraps.get(axis, False):
            relative[axis] = 1.0
        elif denominator > 0:
            relative[axis] = min(float(extents[axis] / denominator), 1.0)
    return extents, relative, wraps


def approximate_mean_shortest_path(graph: nx.Graph, samples: int, seed: int) -> float:
    size = graph.number_of_nodes()
    if size < 2:
        return 0.0 if size == 1 else np.nan
    nodes = list(graph)
    rng = np.random.default_rng(seed)
    sources = nodes if size <= samples else [nodes[i] for i in rng.choice(size, size=samples, replace=False)]
    total = 0
    count = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(graph, source)
        total += sum(length for node, length in lengths.items() if node != source)
        count += len(lengths) - 1
    return float(total / count) if count else np.nan


def mean_betweenness(graph: nx.Graph, samples: int, seed: int) -> float:
    size = graph.number_of_nodes()
    if size < 3:
        return 0.0
    k = None if size <= samples else samples
    values = nx.betweenness_centrality(graph, k=k, normalized=True, seed=seed)
    return float(np.mean(list(values.values()))) if values else 0.0


def widest_path_threshold(
    graph: nx.Graph,
    bottom_nodes: set,
    top_nodes: set,
    mean_force: float,
) -> float:
    sources = set(graph).intersection(bottom_nodes)
    targets = set(graph).intersection(top_nodes)
    if not sources or not targets or mean_force <= 0:
        return np.nan
    capacity = {node: -np.inf for node in graph}
    heap = []
    for node in sources:
        capacity[node] = np.inf
        heapq.heappush(heap, (-capacity[node], str(node), node))
    while heap:
        negative, _, node = heapq.heappop(heap)
        value = -negative
        if value < capacity[node]:
            continue
        if node in targets:
            return float(value)
        for neighbor, data in graph[node].items():
            edge_capacity = float(data.get("normal_force", 0.0)) / mean_force
            candidate = min(value, edge_capacity)
            if candidate > capacity[neighbor]:
                capacity[neighbor] = candidate
                heapq.heappush(heap, (-candidate, str(neighbor), neighbor))
    return np.nan


def split_periodic_segment(
    start: np.ndarray,
    delta: np.ndarray,
    box_lengths: dict[int, float],
    periodic_axes: list[int],
) -> list[np.ndarray]:
    current = np.asarray(start, dtype=float).copy()
    remaining = np.asarray(delta, dtype=float).copy()
    pieces = []
    for _ in range(len(periodic_axes) + 1):
        endpoint = current + remaining
        crossings = []
        for axis in periodic_axes:
            length = box_lengths[axis]
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


def plot_four_views(
    graph: nx.Graph,
    strong_edges: set,
    largest_nodes: set,
    box_lengths: dict[int, float],
    periodic_axes: list[int],
    title: str,
    output: Path,
) -> None:
    nodes = list(graph)
    xyz = np.asarray([graph.nodes[node]["position"] for node in nodes], dtype=float)
    index = {node: idx for idx, node in enumerate(nodes)}
    largest_segments, other_segments, weak_segments = [], [], []
    for u, v in graph.edges():
        start = xyz[index[u]]
        delta = minimum_image(xyz[index[v]] - start, box_lengths, periodic_axes)
        edge = frozenset((u, v))
        if edge in strong_edges and u in largest_nodes and v in largest_nodes:
            target = largest_segments
        elif edge in strong_edges:
            target = other_segments
        else:
            target = weak_segments
        if periodic_axes:
            target.extend(split_periodic_segment(start, delta, box_lengths, periodic_axes))
        else:
            target.append(np.vstack((start, start + delta)))
    largest_mask = np.asarray([node in largest_nodes for node in nodes])
    strong_nodes = set().union(*(set(edge) for edge in (tuple(item) for item in strong_edges))) if strong_edges else set()
    other_mask = np.asarray([node in strong_nodes and node not in largest_nodes for node in nodes])
    weak_mask = ~(largest_mask | other_mask)
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    span = np.maximum(maxs - mins, np.finfo(float).eps)
    fig = plt.figure(figsize=(12, 10))
    axes = [fig.add_subplot(2, 2, idx + 1, projection="3d") for idx in range(4)]
    for axis, (elev, azim, view_title) in zip(axes, VIEWS):
        if weak_segments:
            axis.add_collection3d(Line3DCollection(weak_segments, colors="#A8A8A8", linewidths=0.22, alpha=0.14, rasterized=True))
        if other_segments:
            axis.add_collection3d(Line3DCollection(other_segments, colors="#1F77B4", linewidths=0.65, alpha=0.72, rasterized=True))
        if largest_segments:
            axis.add_collection3d(Line3DCollection(largest_segments, colors="#D62728", linewidths=1.2, alpha=0.95, rasterized=True))
        axis.scatter(xyz[weak_mask, 0], xyz[weak_mask, 1], xyz[weak_mask, 2], s=1.4, c="#A8A8A8", alpha=0.22, depthshade=False, rasterized=True)
        axis.scatter(xyz[other_mask, 0], xyz[other_mask, 1], xyz[other_mask, 2], s=4, c="#1F77B4", alpha=0.75, depthshade=False, rasterized=True)
        axis.scatter(xyz[largest_mask, 0], xyz[largest_mask, 1], xyz[largest_mask, 2], s=6, c="#D62728", alpha=0.95, depthshade=False, rasterized=True)
        axis.view_init(elev=elev, azim=azim)
        axis.set_proj_type("persp" if view_title == "Perspective 3D" else "ortho")
        axis.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]), zlim=(mins[2], maxs[2]), title=view_title)
        axis.set_box_aspect(span)
        axis.set_axis_off()
    fig.suptitle(title)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#D62728", lw=1.5, label="Largest strong cluster"),
            Line2D([0], [0], color="#1F77B4", lw=1.1, label="Other strong clusters"),
            Line2D([0], [0], color="#A8A8A8", lw=1.0, label="Below-threshold contacts"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.07, top=0.92, wspace=0.02, hspace=0.05)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_threshold(
    core: nx.Graph,
    n_value: float,
    mean_force: float,
    bottom_nodes: set,
    top_nodes: set,
    all_spans: np.ndarray,
    box_lengths: dict[int, float],
    periodic_axes: list[int],
    cfg: dict,
    seed: int,
) -> tuple[dict, pd.DataFrame, set, set]:
    threshold = n_value * mean_force
    edges = [(u, v) for u, v, data in core.edges(data=True) if float(data.get("normal_force", 0.0)) >= threshold]
    strong = nx.Graph()
    strong.add_edges_from(edges)
    for node in strong:
        strong.nodes[node].update(core.nodes[node])
    for u, v in strong.edges():
        strong.edges[u, v].update(core.edges[u, v])
    components = sorted(nx.connected_components(strong), key=lambda nodes: (-len(nodes), min(map(str, nodes))))
    component_rows = []
    largest = strong.subgraph(components[0]).copy() if components else nx.Graph()
    for component_id, nodes in enumerate(components):
        cluster = strong.subgraph(nodes).copy()
        extents, relative, wraps = component_lengths(cluster, all_spans, box_lengths, periodic_axes)
        degrees = [degree for _, degree in cluster.degree()]
        row = {
            "n_value": n_value,
            "threshold_force": threshold,
            "component_id": component_id,
            "is_largest": component_id == 0,
            "node_count": cluster.number_of_nodes(),
            "edge_count": cluster.number_of_edges(),
            "mean_degree": float(np.mean(degrees)) if degrees else np.nan,
            "clustering_coefficient": float(np.mean(list(nx.clustering(cluster).values()))) if cluster else np.nan,
            "touches_bottom": bool(set(nodes) & bottom_nodes),
            "touches_top": bool(set(nodes) & top_nodes),
            "spans_loading": bool(set(nodes) & bottom_nodes and set(nodes) & top_nodes),
            "extent_x": extents[0],
            "extent_y": extents[1],
            "extent_z": extents[2],
            "relative_length_x": relative[0],
            "relative_length_y": relative[1],
            "relative_length_z": relative[2],
            "wraps_x": bool(wraps.get(0, False)),
            "wraps_y": bool(wraps.get(1, False)),
        }
        component_rows.append(row)
    component_frame = pd.DataFrame(component_rows)
    sizes = [len(nodes) for nodes in components]
    largest_size = sizes[0] if sizes else 0
    second_size = sizes[1] if len(sizes) > 1 else 0
    extents, relative, wraps = component_lengths(largest, all_spans, box_lengths, periodic_axes)
    if largest.number_of_nodes() > 1:
        diameter = float(nx.diameter(largest, usebounds=True))
        radius = float(nx.radius(largest, usebounds=True))
    elif largest.number_of_nodes() == 1:
        diameter = radius = 0.0
    else:
        diameter = radius = np.nan
    degrees = [degree for _, degree in strong.degree()]
    mean_degree = float(np.mean(degrees)) if degrees else np.nan
    clustering = float(np.mean(list(nx.clustering(strong).values()))) if strong else np.nan
    betweenness = mean_betweenness(strong, int(cfg["betweenness_samples"]), seed)
    divisor = np.cbrt(core.number_of_nodes())
    largest_nodes = set(largest)
    row = {
        "n_value": n_value,
        "threshold_force": threshold,
        "mean_normal_force": mean_force,
        "total_particle_count": core.number_of_nodes(),
        "total_contact_count": core.number_of_edges(),
        "strong_particle_count": strong.number_of_nodes(),
        "strong_contact_count": strong.number_of_edges(),
        "strong_particle_fraction": strong.number_of_nodes() / core.number_of_nodes() if core else np.nan,
        "strong_contact_fraction": strong.number_of_edges() / core.number_of_edges() if core.number_of_edges() else np.nan,
        "cluster_count": len(components),
        "largest_cluster_size": largest_size,
        "second_largest_cluster_size": second_size,
        "second_to_largest_ratio": second_size / largest_size if largest_size else np.nan,
        "largest_cluster_fraction": largest_size / core.number_of_nodes() if core else np.nan,
        "relative_length_x": relative[0],
        "relative_length_y": relative[1],
        "relative_length_z": relative[2],
        "largest_wraps_x": bool(wraps.get(0, False)),
        "largest_wraps_y": bool(wraps.get(1, False)),
        "largest_touches_bottom": bool(largest_nodes & bottom_nodes),
        "largest_touches_top": bool(largest_nodes & top_nodes),
        "largest_spans_loading": bool(largest_nodes & bottom_nodes and largest_nodes & top_nodes),
        "any_cluster_spans_loading": any(bool(set(nodes) & bottom_nodes and set(nodes) & top_nodes) for nodes in components),
        "mean_degree": mean_degree,
        "clustering_coefficient": clustering,
        "largest_diameter": diameter,
        "largest_radius": radius,
        "largest_diameter_normalized": diameter / divisor if np.isfinite(diameter) else np.nan,
        "largest_radius_normalized": radius / divisor if np.isfinite(radius) else np.nan,
        "largest_mean_shortest_path": approximate_mean_shortest_path(largest, int(cfg["shortest_path_samples"]), seed),
        "mean_betweenness": betweenness,
    }
    return row, component_frame, {frozenset(edge) for edge in edges}, largest_nodes


def run_task(
    cfg: dict,
    config_path: Path,
    task: tuple[str, str, int],
    values: list[float],
    output_override: Path | None = None,
    skip_visualizations: bool = False,
) -> None:
    dataset_key, angle, sim_idx = task
    dataset = cfg["datasets"][dataset_key]
    output_root = (output_override / dataset_key if output_override else Path(dataset["output_root"]))
    manifest_path = output_root / "artifacts" / "task_manifests" / f"{angle}_sim{sim_idx:03d}.json"
    task_table = output_root / "artifacts" / "task_summaries" / f"{angle}_sim{sim_idx:03d}_sweep.csv"
    current_fingerprint = fingerprint(cfg, config_path)
    if manifest_path.exists() and task_table.exists() and not output_override:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") == current_fingerprint and manifest.get("n_values") == values:
            print(f"Skipping valid {dataset_key} {angle} simulation {sim_idx}")
            return
    graphs = load_graphs(dataset["graph_pickle"])
    core = graphs[angle]["core"][sim_idx]
    full = graphs[angle]["full"][sim_idx]
    forces = np.asarray([data.get("normal_force", np.nan) for _, _, data in core.edges(data=True)], dtype=float)
    forces = forces[np.isfinite(forces)]
    if not len(forces):
        raise ValueError(f"No finite particle-particle forces for {dataset_key} {angle} simulation {sim_idx}")
    mean_force = float(np.mean(forces))
    bottom_nodes = wall_contact_particles(full, dataset["bottom_wall_labels"])
    top_nodes = wall_contact_particles(full, dataset["top_wall_labels"])
    if not bottom_nodes or not top_nodes:
        raise ValueError(f"Missing top/bottom wall-contact particles for {dataset_key} {angle} simulation {sim_idx}")
    positions = np.asarray([data["position"] for _, data in core.nodes(data=True)], dtype=float)
    all_spans = np.ptp(positions, axis=0)
    exact_threshold = widest_path_threshold(core, bottom_nodes, top_nodes, mean_force)
    rows = []
    decimals = int(cfg["sweep"]["decimals"])
    base_seed = int(cfg["random_seed"]) + sim_idx + 1000 * list(cfg["datasets"]).index(dataset_key) + 100 * dataset["angles"].index(angle)
    for value_index, n_value in enumerate(values):
        row, components, strong_edges, largest_nodes = analyze_threshold(
            core,
            n_value,
            mean_force,
            bottom_nodes,
            top_nodes,
            all_spans,
            dataset["box_lengths"],
            dataset["periodic_axes"],
            cfg,
            base_seed + value_index,
        )
        row.update(dataset=dataset_key, angle=angle, sim_idx=sim_idx, exact_any_spanning_n=exact_threshold)
        rows.append(row)
        n_root = output_root / "sweep" / n_folder(n_value, decimals)
        component_columns = ["dataset", "angle", "sim_idx"]
        components.insert(0, "sim_idx", sim_idx)
        components.insert(0, "angle", angle)
        components.insert(0, "dataset", dataset_key)
        atomic_csv(n_root / "tables" / f"{angle}_sim{sim_idx:03d}_components.csv", components)
        atomic_csv(n_root / "tables" / f"{angle}_sim{sim_idx:03d}_network.csv", pd.DataFrame([row]))
        if not skip_visualizations:
            plot_four_views(
                core,
                strong_edges,
                largest_nodes,
                dataset["box_lengths"],
                dataset["periodic_axes"],
                f"{dataset['label']}: {angle}, simulation {sim_idx}, n={n_value:.{decimals}f}",
                n_root / "visualizations" / angle / f"sim{sim_idx:03d}_four_views.png",
            )
    frame = pd.DataFrame(rows)
    baseline_rows = frame.loc[np.isclose(frame.n_value, 0), "mean_betweenness"]
    if len(baseline_rows):
        baseline = float(baseline_rows.iloc[0])
    else:
        baseline = mean_betweenness(core, int(cfg["betweenness_samples"]), base_seed)
    frame["betweenness_ratio"] = frame["mean_betweenness"] / baseline if baseline > 0 else np.nan
    # Refresh per-n network files so they include the within-system B_n/B_0 ratio.
    for _, row in frame.iterrows():
        n_root = output_root / "sweep" / n_folder(float(row.n_value), decimals)
        atomic_csv(n_root / "tables" / f"{angle}_sim{sim_idx:03d}_network.csv", pd.DataFrame([row]))
    atomic_csv(task_table, frame)
    atomic_json(
        manifest_path,
        {
            "dataset": dataset_key,
            "angle": angle,
            "sim_idx": sim_idx,
            "n_values": values,
            "mean_particle_particle_normal_force": mean_force,
            "exact_any_spanning_n": exact_threshold,
            "criterion": "largest strong particle-particle cluster touches bottom and top wall-contact particle sets",
            "fingerprint": current_fingerprint,
        },
    )
    print(f"Complete {dataset_key} {angle} simulation {sim_idx}: {len(values)} thresholds")


def ci_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    mean = float(np.mean(values)) if count else np.nan
    std = float(np.std(values, ddof=1)) if count > 1 else np.nan
    sem = std / np.sqrt(count) if count > 1 else np.nan
    half = float(stats.t.ppf(0.975, count - 1) * sem) if count > 1 else np.nan
    return {
        "count": count,
        "mean": mean,
        "median": float(np.median(values)) if count else np.nan,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half if count > 1 else np.nan,
        "ci95_high": mean + half if count > 1 else np.nan,
    }


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    df = len(a) + len(b) - 2
    pooled = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / df
    if pooled <= 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / np.sqrt(pooled) * (1 - 3 / (4 * df - 1)))


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return a stable two-sided Mann-Whitney result, including tied constants."""
    if not len(a) or not len(b):
        return np.nan, np.nan
    if np.all(np.concatenate((a, b)) == a[0]):
        return float(len(a) * len(b) / 2), 1.0
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def select_thresholds(sweep: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, angle, sim_idx), frame in sweep.groupby(["dataset", "angle", "sim_idx"], sort=False):
        frame = frame.sort_values("n_value")
        spanning = frame.loc[frame["largest_spans_loading"].astype(bool), "n_value"]
        selected = float(spanning.max()) if len(spanning) else np.nan
        diameter_row = frame.loc[frame["largest_diameter_normalized"].idxmax()] if frame["largest_diameter_normalized"].notna().any() else None
        cluster_row = frame.loc[frame["cluster_count"].idxmax()] if frame["cluster_count"].notna().any() else None
        chained = frame.loc[frame["largest_spans_loading"].astype(bool)].copy()
        chain_n = np.nan
        if len(chained):
            chain_n = float(chained.loc[(chained["mean_degree"] - 2.0).abs().idxmin(), "n_value"])
        records.append(
            {
                "dataset": dataset,
                "angle": angle,
                "sim_idx": sim_idx,
                "selected_n_largest_spanning": selected,
                "exact_any_spanning_n": float(frame["exact_any_spanning_n"].iloc[0]),
                "n_at_maximum_diameter": float(diameter_row.n_value) if diameter_row is not None else np.nan,
                "maximum_normalized_diameter": float(diameter_row.largest_diameter_normalized) if diameter_row is not None else np.nan,
                "n_at_maximum_cluster_count": float(cluster_row.n_value) if cluster_row is not None else np.nan,
                "maximum_cluster_count": int(cluster_row.cluster_count) if cluster_row is not None else 0,
                "n_closest_to_degree_2_while_spanning": chain_n,
            }
        )
    return pd.DataFrame(records)


def curve_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, frame in sweep.groupby(["dataset", "angle", "n_value"], sort=False):
        dataset, angle, n_value = keys
        for prop in CURVE_PROPERTIES:
            summary = ci_summary(pd.to_numeric(frame[prop], errors="coerce").to_numpy(float))
            records.append({"dataset": dataset, "angle": angle, "n_value": n_value, "property": prop, **summary})
    return pd.DataFrame(records)


def plot_curves(curves: pd.DataFrame, cfg: dict, output_root: Path, datasets: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    colors = {"0deg": "#1F77B4", "30deg": "#FF7F0E"}
    for prop, ylabel in CURVE_PROPERTIES.items():
        fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.5), squeeze=False, sharey=False)
        for axis, dataset_key in zip(axes[0], datasets):
            dataset_cfg = cfg["datasets"][dataset_key]
            for angle in dataset_cfg["angles"]:
                subset = curves[(curves.dataset == dataset_key) & (curves.angle == angle) & (curves.property == prop)].sort_values("n_value")
                x = subset.n_value.to_numpy(float)
                mean = subset["mean"].to_numpy(float)
                low = subset.ci95_low.to_numpy(float)
                high = subset.ci95_high.to_numpy(float)
                axis.plot(x, mean, color=colors[angle], label=angle)
                axis.fill_between(x, low, high, color=colors[angle], alpha=0.18, linewidth=0)
            axis.axvline(1.5, color="#555555", linestyle="--", linewidth=1, label="paper ~1.5" if dataset_key == datasets[0] else None)
            axis.set(title=dataset_cfg["label"], xlabel=r"Threshold multiplier $n$")
            axis.grid(alpha=0.2)
        axes[0, 0].set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.subplots_adjust(bottom=0.18, wspace=0.25)
        fig.savefig(output_root / f"{prop}_mean_ci.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_selected_thresholds(
    selected: pd.DataFrame,
    cfg: dict,
    output_root: Path,
    datasets: list[str],
    include_combined: bool = True,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    colors = {"0deg": "#1F77B4", "30deg": "#FF7F0E"}
    selected_here = selected[selected.dataset.isin(datasets)]
    all_values = selected_here["selected_n_largest_spanning"].dropna().to_numpy(float)
    lo, hi = (all_values.min(), all_values.max()) if len(all_values) else (0, 1)
    step = float(cfg["sweep"]["step"])
    edges = np.arange(lo - step / 2, hi + 1.5 * step, step)
    if len(edges) < 2:
        edges = np.asarray([lo - 0.5, hi + 0.5])
    for dataset_key in datasets:
        subset = selected[selected.dataset == dataset_key]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for angle in cfg["datasets"][dataset_key]["angles"]:
            values = subset.loc[subset.angle == angle, "selected_n_largest_spanning"].dropna().to_numpy(float)
            axes[0].hist(
                values,
                bins=edges,
                density=True,
                histtype="step",
                label=angle,
                color=colors[angle],
                linewidth=1.7,
            )
        axes[0].set(xlabel="Selected critical n", ylabel="Density", title="Distribution (shared bins)")
        groups = [subset.loc[subset.angle == angle, "selected_n_largest_spanning"].dropna().to_numpy(float) for angle in cfg["datasets"][dataset_key]["angles"]]
        labels = cfg["datasets"][dataset_key]["angles"]
        axes[1].boxplot(groups, labels=labels, showmeans=True)
        rng = np.random.default_rng(int(cfg["random_seed"]))
        for idx, (values, color) in enumerate(zip(groups, [colors[label] for label in labels]), start=1):
            axes[1].scatter(idx + rng.uniform(-0.06, 0.06, len(values)), values, s=18, color=color, alpha=0.75)
        axes[1].set(ylabel="Selected critical n", title="Simulation values")
        for axis in axes:
            axis.grid(alpha=0.2)
        axes[0].legend(frameon=False)
        fig.suptitle(cfg["datasets"][dataset_key]["label"])
        fig.tight_layout()
        fig.savefig(output_root / f"{dataset_key}_selected_n_distribution_boxplot.png", dpi=220)
        plt.close(fig)
    if include_combined:
        labels = []
        groups = []
        for dataset_key in datasets:
            for angle in cfg["datasets"][dataset_key]["angles"]:
                labels.append(f"{cfg['datasets'][dataset_key]['short_label']}\n{angle}")
                groups.append(selected.loc[(selected.dataset == dataset_key) & (selected.angle == angle), "selected_n_largest_spanning"].dropna().to_numpy(float))
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.boxplot(groups, labels=labels, showmeans=True)
        for idx, values in enumerate(groups, start=1):
            axis.scatter(np.full(len(values), idx), values, s=16, alpha=0.62)
        axis.set(ylabel="Selected critical n", title="Critical force-threshold comparison")
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_root / "all_datasets_selected_n_boxplot.png", dpi=220)
        plt.close(fig)


def comparison_tables(selected: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    for (dataset, angle), frame in selected.groupby(["dataset", "angle"], sort=False):
        summaries.append({"dataset": dataset, "angle": angle, **ci_summary(frame["selected_n_largest_spanning"].to_numpy(float))})
    tests = []
    for dataset_key, dataset in cfg["datasets"].items():
        paired_frame = selected.loc[
            selected.dataset == dataset_key,
            ["sim_idx", "angle", "selected_n_largest_spanning"],
        ].pivot(index="sim_idx", columns="angle", values="selected_n_largest_spanning")
        a = paired_frame["0deg"].to_numpy(float)
        b = paired_frame["30deg"].to_numpy(float)
        paired_mask = np.isfinite(a) & np.isfinite(b)
        paired_a, paired_b = a[paired_mask], b[paired_mask]
        finite_a, finite_b = a[np.isfinite(a)], b[np.isfinite(b)]
        welch = stats.ttest_ind(finite_a, finite_b, equal_var=False) if len(finite_a) > 1 and len(finite_b) > 1 else None
        mann_u, mann_p = mann_whitney(finite_a, finite_b)
        paired = stats.ttest_rel(paired_a, paired_b) if len(paired_a) > 1 else None
        diff = paired_b - paired_a
        diff_ci = ci_summary(diff)
        tests.append({
            "comparison": f"{dataset_key}: 30deg - 0deg",
            "mean_difference": float(np.mean(finite_b) - np.mean(finite_a)) if len(finite_a) and len(finite_b) else np.nan,
            "paired_mean_difference": diff_ci["mean"],
            "paired_ci95_low": diff_ci["ci95_low"],
            "paired_ci95_high": diff_ci["ci95_high"],
            "welch_t": float(welch.statistic) if welch else np.nan,
            "welch_p": float(welch.pvalue) if welch else np.nan,
            "paired_t": float(paired.statistic) if paired else np.nan,
            "paired_p": float(paired.pvalue) if paired else np.nan,
            "mann_whitney_u": mann_u,
            "mann_whitney_p": mann_p,
            "hedges_g_second_minus_first": hedges_g(finite_a, finite_b),
        })
    for angle in ("0deg", "30deg"):
        keys = list(cfg["datasets"])
        a = selected.loc[(selected.dataset == keys[0]) & (selected.angle == angle), "selected_n_largest_spanning"].dropna().to_numpy(float)
        b = selected.loc[(selected.dataset == keys[1]) & (selected.angle == angle), "selected_n_largest_spanning"].dropna().to_numpy(float)
        welch = stats.ttest_ind(a, b, equal_var=False) if len(a) > 1 and len(b) > 1 else None
        tests.append({
            "comparison": f"{angle}: {keys[1]} - {keys[0]}",
            "mean_difference": float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan,
            "welch_t": float(welch.statistic) if welch else np.nan,
            "welch_p": float(welch.pvalue) if welch else np.nan,
            "hedges_g_second_minus_first": hedges_g(a, b),
        })
    return pd.DataFrame(summaries), pd.DataFrame(tests)


def expose_selected_views(selected: pd.DataFrame, cfg: dict) -> None:
    decimals = int(cfg["sweep"]["decimals"])
    for dataset_key, dataset in cfg["datasets"].items():
        destination = Path(dataset["output_root"]) / "summary" / "selected_cluster_visualizations"
        destination.mkdir(parents=True, exist_ok=True)
        for _, row in selected[selected.dataset == dataset_key].iterrows():
            if not np.isfinite(row.selected_n_largest_spanning):
                continue
            source = Path(dataset["output_root"]) / "sweep" / n_folder(float(row.selected_n_largest_spanning), decimals) / "visualizations" / row.angle / f"sim{int(row.sim_idx):03d}_four_views.png"
            target = destination / f"{row.angle}_sim{int(row.sim_idx):03d}_n_{float(row.selected_n_largest_spanning):.{decimals}f}.png"
            if not source.exists():
                continue
            relative = os.path.relpath(source, target.parent)
            if target.is_symlink() and os.readlink(target) == relative:
                continue
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(relative)


def write_readme(root: Path, dataset: dict, cfg: dict) -> None:
    sweep = cfg["sweep"]
    text = f"""# 3 — Force-threshold percolation

This analysis adapts Liu et al. (2023), *A network-based investigation on
the strong contact system of granular materials under isotropic and deviatoric
stress states*, DOI `10.1016/j.compgeo.2022.105077`.

For each simulation independently, the particle-particle mean normal force is
computed and contacts satisfying

```text
normal_force >= n * mean_particle_particle_normal_force
```

form the strong network. Wall contacts do not enter the mean or the strong
network. The sweep is `n={sweep['start']:.1f}` through `n={sweep['stop']:.1f}`
in increments of `{sweep['step']:.1f}` for 0deg and 30deg.

The primary selected critical value is the largest sampled n for which the
largest strong cluster still connects particles directly contacting the bottom
wall to particles directly contacting the top wall. This is the finite-system,
loading-direction implementation of the paper's boundary-spanning largest-
cluster criterion. The exact widest-path threshold, diameter-peak threshold,
cluster-count-peak threshold, and the spanning threshold closest to mean degree
2 are also retained as diagnostics.

- `sweep/n_*/tables/`: per-simulation network and component characteristics.
- `sweep/n_*/visualizations/`: four-view 3D figures; largest cluster is red,
  other strong clusters blue, and below-threshold contacts gray.
- `summary/curves/`: simulation means and 95% t confidence intervals versus n.
- `summary/critical_thresholds/`: selected-n distributions, boxplots, tables,
  confidence intervals, and tests.
- `summary/selected_cluster_visualizations/`: links to each simulation's
  selected-threshold four-view figure.
- `artifacts/`: restartable task tables and completion manifests.

For periodic samples, particle contacts and visualizations use exact x/y
minimum-image vectors. The percolation decision itself uses the nonperiodic
loading direction z and the physical top/bottom wall-contact labels.
"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(text)


def summarize(cfg: dict) -> None:
    frames = []
    expected = 0
    for dataset_key, dataset in cfg["datasets"].items():
        task_dir = Path(dataset["output_root"]) / "artifacts" / "task_summaries"
        files = sorted(task_dir.glob("*_sweep.csv"))
        dataset_expected = len(dataset["angles"]) * int(dataset["simulations_per_angle"])
        if len(files) != dataset_expected:
            raise RuntimeError(f"{dataset_key}: found {len(files)} task tables; expected {dataset_expected}")
        frames.extend(pd.read_csv(path) for path in files)
        expected += dataset_expected
    sweep = pd.concat(frames, ignore_index=True)
    selected = select_thresholds(sweep)
    curves = curve_summary(sweep)
    group_stats, tests = comparison_tables(selected, cfg)
    for dataset_key, dataset in cfg["datasets"].items():
        root = Path(dataset["output_root"])
        summary_root = root / "summary"
        subset_sweep = sweep[sweep.dataset == dataset_key]
        subset_selected = selected[selected.dataset == dataset_key]
        subset_curves = curves[curves.dataset == dataset_key]
        atomic_csv(summary_root / "tables" / "sweep_metrics.csv", subset_sweep)
        atomic_csv(summary_root / "critical_thresholds" / "selected_thresholds.csv", subset_selected)
        atomic_csv(summary_root / "critical_thresholds" / "group_mean_ci.csv", group_stats[group_stats.dataset == dataset_key])
        atomic_csv(summary_root / "critical_thresholds" / "comparisons.csv", tests[tests.comparison.str.startswith(dataset_key)])
        atomic_csv(summary_root / "tables" / "curve_mean_ci.csv", subset_curves)
        plot_curves(subset_curves, cfg, summary_root / "curves", [dataset_key])
        plot_selected_thresholds(
            subset_selected,
            cfg,
            summary_root / "critical_thresholds",
            [dataset_key],
            include_combined=False,
        )
        write_readme(root, dataset, cfg)
    combined_root = Path(cfg["combined_output_root"]) / "summary"
    atomic_csv(combined_root / "tables" / "all_sweep_metrics.csv", sweep)
    atomic_csv(combined_root / "tables" / "all_selected_thresholds.csv", selected)
    atomic_csv(combined_root / "tables" / "selected_threshold_group_mean_ci.csv", group_stats)
    atomic_csv(combined_root / "tables" / "selected_threshold_comparisons.csv", tests)
    atomic_csv(combined_root / "tables" / "all_curve_mean_ci.csv", curves)
    plot_curves(curves, cfg, combined_root / "curves", list(cfg["datasets"]))
    plot_selected_thresholds(
        selected,
        cfg,
        combined_root / "critical_thresholds",
        list(cfg["datasets"]),
        include_combined=True,
    )
    expose_selected_views(selected, cfg)
    (Path(cfg["combined_output_root"]) / "README.md").write_text(
        "# Force-threshold percolation comparison\n\n"
        "This folder compares final-load and periodic-boundary results from each "
        "dataset's `3_force_threshold_percolation` analysis. The combined summary "
        "contains two-panel structure curves and four-group selected-threshold "
        "comparisons for 0deg and 30deg. See the dataset READMEs for the criterion.\n"
    )
    print(json.dumps({"task_tables": expected, "sweep_rows": len(sweep), "selected_rows": len(selected), "combined_output": str(combined_root)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-values", help="Comma-separated override, intended for validation pilots")
    parser.add_argument("--output-root", type=Path, help="Validation-only output root")
    parser.add_argument("--skip-visualizations", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    tasks = all_tasks(cfg)
    values = threshold_values(cfg, args.n_values)
    if args.summary:
        if args.dry_run:
            print(f"Would summarize {len(tasks)} task tables")
            return
        summarize(cfg)
        return
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None
    if task_id is None or task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"Provide --task-id in 0..{len(tasks)-1}")
    if args.dry_run:
        print({"task_id": task_id, "task": tasks[task_id], "n_values": values})
        return
    run_task(cfg, args.config.resolve(), tasks[task_id], values, args.output_root, args.skip_visualizations)


if __name__ == "__main__":
    main()
