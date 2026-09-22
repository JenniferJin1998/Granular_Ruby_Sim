#!/usr/bin/env python3
"""Create granular-style property visualizations from finalized crystal tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "crystal_reference_config.json"
STRUCTURE_COLORS = {
    "SC": "#2CA02C",
    "BCC": "#9467BD",
    "FCC": "#8C564B",
    "HCP": "#E377C2",
}

# Keep this inventory aligned with the main crystal workflow without importing
# it: importing that module also initializes the graph pipeline's output state.
NODE_PROPERTIES = [
    "degree", "closeness", "betweenness", "clustering",
    "avg_neighbor_degree", "principal_eigenvector", "fiedler",
    "avg_curvature_no_walls", "nfd", "nfd_r2",
    "q4", "q6", "q8", "q10", "q12", "qbar4", "qbar6",
    "w4", "what4", "w6", "what6", "mean_s6",
    "fraction_s6_above_threshold",
]
EDGE_PROPERTIES = [
    "angle_with_zz", "periodic_distance", "edge_connectivity",
    "node_connectivity", "curvature_no_walls", "s6",
]
GRAPH_PROPERTIES = [
    "num_nodes", "num_edges", "assortativity", "edge_connectivity_graph",
    "node_connectivity_graph", "num_components", "path_nodes", "avg_path",
    "diameter", "radius", "spectral_radius", "alg_connectivity",
    "fiedler_value", "loop_total", "loop_mean", "nfd_asymmetry",
    "nfd_tau_mean", "nfd_tau_std", "Q4", "Q6", "Q8", "Q10", "Q12",
    "mean_local_q4", "mean_local_q6", "mean_local_q8", "mean_local_q10",
    "mean_local_q12",
]


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    root = Path(cfg.get("project_root") or PROJECT_ROOT).resolve()
    cfg["project_root"] = str(root)
    cfg["output_root"] = str((root / cfg["output_root"]).resolve())
    return cfg


def finite(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return array[np.isfinite(array)]


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value).strip("_")


def boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def shared_edges(groups: list[np.ndarray], bins: int) -> np.ndarray | None:
    clean = [values[np.isfinite(values)] for values in groups if len(values)]
    if not clean:
        return None
    pooled = np.concatenate(clean)
    minimum = float(np.min(pooled))
    maximum = float(np.max(pooled))
    if np.isclose(minimum, maximum):
        return None
    return np.linspace(minimum, maximum, bins + 1)


def available(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    return [property_name for property_name in requested if property_name in frame and finite(frame[property_name]).size]


def plot_distribution_panels(
    frame: pd.DataFrame,
    property_name: str,
    structures: list[str],
    bins: int,
    output: Path,
) -> None:
    groups = {structure: finite(frame.loc[frame.geometry == structure, property_name]) for structure in structures}
    edges = shared_edges(list(groups.values()), bins)
    densities = {}
    if edges is not None:
        for structure, values in groups.items():
            densities[structure] = np.histogram(values, bins=edges, density=True)[0] if len(values) else np.asarray([])
    ymax = max((float(np.max(values)) for values in densities.values() if len(values)), default=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True, sharey=True)
    for axis, structure in zip(axes.flat, structures):
        values = groups[structure]
        if edges is None:
            if len(values):
                axis.axvline(float(np.mean(values)), color=STRUCTURE_COLORS[structure], linewidth=2)
                axis.text(0.04, 0.90, f"constant = {np.mean(values):.5g}", transform=axis.transAxes)
        elif len(values):
            axis.step(edges[:-1], densities[structure], where="post", color=STRUCTURE_COLORS[structure], linewidth=1.7)
            axis.axvline(float(np.mean(values)), color="#222222", linestyle="--", linewidth=1.0, label="mean")
            axis.set_xlim(edges[0], edges[-1])
            axis.set_ylim(0, ymax * 1.08)
        axis.set_title(f"{structure} (N={len(values):,})")
        axis.grid(alpha=0.18)
    fig.suptitle(f"Crystal distributions: {property_name.replace('_', ' ')}")
    fig.supxlabel(property_name.replace("_", " "))
    fig.supylabel("Density")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_distribution_boxplot(
    frame: pd.DataFrame,
    property_name: str,
    structures: list[str],
    output: Path,
) -> None:
    groups = [finite(frame.loc[frame.geometry == structure, property_name]) for structure in structures]
    if not any(len(values) for values in groups):
        return
    fig, axis = plt.subplots(figsize=(7.8, 5.0))
    artists = axis.boxplot(groups, labels=structures, showmeans=True, showfliers=False, patch_artist=True)
    for box, structure in zip(artists["boxes"], structures):
        box.set_facecolor(STRUCTURE_COLORS[structure])
        box.set_alpha(0.46)
    for index, values in enumerate(groups, start=1):
        if len(values):
            axis.text(index, 0.02, f"N={len(values):,}", transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    axis.set(title=f"Pooled node/contact values: {property_name.replace('_', ' ')}", ylabel=property_name.replace("_", " "))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_graph_bars(
    frame: pd.DataFrame,
    property_name: str,
    structures: list[str],
    output: Path,
) -> None:
    values = []
    for structure in structures:
        group = finite(frame.loc[frame.geometry == structure, property_name])
        values.append(float(group[0]) if len(group) else np.nan)
    if not np.isfinite(values).any():
        return
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    bars = axis.bar(structures, values, color=[STRUCTURE_COLORS[item] for item in structures], alpha=0.78)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.5g}", ha="center", va="bottom", fontsize=8)
    axis.set(title=f"Whole-graph property: {property_name.replace('_', ' ')}", ylabel=property_name.replace("_", " "))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def property_mean_table(frame: pd.DataFrame, properties: list[str], structures: list[str]) -> pd.DataFrame:
    rows = []
    for property_name in properties:
        row = {"property": property_name}
        for structure in structures:
            values = finite(frame.loc[frame.geometry == structure, property_name])
            row[structure] = float(np.mean(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_standardized_heatmap(table: pd.DataFrame, structures: list[str], domain: str, output: Path) -> None:
    if table.empty:
        return
    matrix = table[structures].to_numpy(float)
    means = np.nanmean(matrix, axis=1, keepdims=True)
    standard_deviations = np.nanstd(matrix, axis=1, keepdims=True)
    standardized = np.divide(matrix - means, standard_deviations, out=np.zeros_like(matrix), where=standard_deviations > 0)
    height = max(5.0, 0.34 * len(table))
    fig, axis = plt.subplots(figsize=(7.8, height))
    image = axis.imshow(standardized, aspect="auto", cmap="coolwarm", vmin=-1.7, vmax=1.7)
    axis.set_xticks(np.arange(len(structures)), labels=structures)
    axis.set_yticks(np.arange(len(table)), labels=table.property.str.replace("_", " "))
    axis.set_title(f"Relative {domain}-property means across crystal structures\n(row-wise z score; color compares structures within one property)")
    fig.colorbar(image, ax=axis, label="Z score among SC/BCC/FCC/HCP")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    output_root = Path(cfg["output_root"])
    data_root = output_root / "0_graph_and_basic_stats" / "graph_data"
    result_root = output_root / "1_network_property_comparison"
    required = {
        "node": data_root / "node_features.csv",
        "edge": data_root / "edge_features.csv",
        "graph": data_root / "graph_features.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Finalized feature tables are missing: " + ", ".join(missing))
    nodes = pd.read_csv(required["node"])
    edges = pd.read_csv(required["edge"])
    graphs = pd.read_csv(required["graph"])
    if "is_core_edge" in edges:
        edges = edges.loc[boolean_mask(edges["is_core_edge"])].copy()
    structures = [str(value) for value in cfg["structures"]]
    bins = int(cfg["histogram_bins"])
    node_properties = available(nodes, NODE_PROPERTIES)
    edge_properties = available(edges, EDGE_PROPERTIES)
    graph_properties = available(graphs, GRAPH_PROPERTIES)

    inventory = []
    mean_tables = []
    for domain, frame, properties in (("node", nodes, node_properties), ("edge", edges, edge_properties)):
        for property_name in properties:
            plot_distribution_panels(
                frame, property_name, structures, bins,
                result_root / "distribution_panels" / domain / f"{safe_name(property_name)}.png",
            )
            plot_distribution_boxplot(
                frame, property_name, structures,
                result_root / "distribution_boxplots" / domain / f"{safe_name(property_name)}.png",
            )
            inventory.append({"domain": domain, "property": property_name, "visualizations": "overlay;four_panel;pooled_boxplot"})
        means = property_mean_table(frame, properties, structures)
        means.insert(0, "domain", domain)
        mean_tables.append(means)
        plot_standardized_heatmap(
            means.drop(columns="domain"), structures, domain,
            result_root / "summary_heatmaps" / f"{domain}_property_means_zscore.png",
        )

    for property_name in graph_properties:
        plot_graph_bars(
            graphs, property_name, structures,
            result_root / "graph_property_bars" / f"{safe_name(property_name)}.png",
        )
        inventory.append({"domain": "graph", "property": property_name, "visualizations": "overlay;bar"})
    graph_means = property_mean_table(graphs, graph_properties, structures)
    graph_means.insert(0, "domain", "graph")
    mean_tables.append(graph_means)
    plot_standardized_heatmap(
        graph_means.drop(columns="domain"), structures, "graph",
        result_root / "summary_heatmaps" / "graph_property_values_zscore.png",
    )

    table_root = result_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(table_root / "visualized_property_inventory.csv", index=False)
    pd.concat(mean_tables, ignore_index=True).to_csv(table_root / "property_means_by_crystal.csv", index=False)
    (result_root / "PROPERTY_VISUALIZATIONS.md").write_text(
        "# Crystal property visualizations\n\n"
        "- `distributions/` contains shared-bin overlays produced by the main finalizer.\n"
        "- `distribution_panels/` shows the same node/contact distributions in four panels with shared x/y limits.\n"
        "- `distribution_boxplots/` compares pooled node/contact values; these boxes are not independent simulation replicates.\n"
        "- `graph_property_bars/` compares the one whole-graph value available for each deterministic crystal.\n"
        "- `summary_heatmaps/` standardizes each property across SC/BCC/FCC/HCP to emphasize relative differences.\n"
        "- `boundary_layers/` retains the all-surfaces, top-only, and bottom-only distance analyses.\n"
        "- `tables/` stores descriptive summaries, plotted-property inventory, and crystal means.\n"
    )
    print(
        f"Created granular-style crystal plots for {len(node_properties)} node, "
        f"{len(edge_properties)} edge, and {len(graph_properties)} graph properties"
    )


if __name__ == "__main__":
    main()
