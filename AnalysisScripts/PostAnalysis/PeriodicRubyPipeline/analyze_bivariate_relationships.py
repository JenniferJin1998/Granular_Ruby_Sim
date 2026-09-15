#!/usr/bin/env python3
"""Bivariate PBC property, force, and bond-order comparisons.

The script consumes the corrected Job 4/5 tables. Two-dimensional bins are
normalized to probability mass within each panel and use shared axes, bin
edges, and logarithmic color limits across every population being compared.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import networkx as nx
import numpy as np
import pandas as pd

from crystal_common import STRUCTURES, crystal_graph
from pipeline_common import atomic_csv, atomic_json, finite, load_config, replicate_test


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config_periodic_corrected.yaml"
Q_PROPERTIES = ("q4", "q6", "qbar4", "qbar6")
NODE_PROPERTIES = (
    "z", "degree", "clustering", "avg_neighbor_degree",
    "avg_curvature_no_walls", "betweenness", "closeness", "fiedler",
    "high_force_degree", "nfd", "nfd_r2", "principal_eigenvector",
)
EDGE_PROPERTIES = ("angle_with_zz", "curvature_no_walls")
LABELS = {
    "stress_hydro": "Hydrostatic stress",
    "normal_force": "Normal force",
    "z": "z position",
    "degree": "Degree",
    "clustering": "Clustering coefficient",
    "avg_neighbor_degree": "Average neighbor degree",
    "avg_curvature_no_walls": "Average curvature (no walls)",
    "curvature_no_walls": "Curvature (no walls)",
    "angle_with_zz": "Contact angle with z (degrees)",
    "betweenness": "Betweenness centrality",
    "closeness": "Closeness centrality",
    "fiedler": "Fiedler-vector component",
    "high_force_degree": "High-force degree",
    "nfd": "Node fractal dimension",
    "nfd_r2": "Node fractal-dimension R²",
    "principal_eigenvector": "Principal eigenvector centrality",
    "q4": r"$q_4$", "q6": r"$q_6$",
    "qbar4": r"$\bar q_4$", "qbar6": r"$\bar q_6$",
}
GROUP_COLORS = {"non_high_force": "#2474B5", "high_force": "#D95F02"}


def label(name):
    return LABELS.get(name, name.replace("_", " ").title())


def safe_values(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def load_tables(cfg):
    root = Path(cfg["output_root"])
    node_frames, edge_frames = [], []
    pattern = re.compile(r"ruby_(.+)_sim(\d+)_nodes\.csv$")
    for q_path in sorted((root / "job4_bond_order").glob("ruby_*_nodes.csv")):
        match = pattern.match(q_path.name)
        if not match:
            continue
        angle, sim_text = match.groups()
        sim_idx = int(sim_text)
        node_path = root / "job5_high_force_comparison" / f"{angle}_sim{sim_idx:03d}_complete_nodes.csv"
        edge_path = root / "job5_high_force_comparison" / f"{angle}_sim{sim_idx:03d}_complete_edges.csv"
        if not node_path.exists() or not edge_path.exists():
            raise FileNotFoundError(f"Missing corrected Job 5 tables for {angle} simulation {sim_idx}")
        q = pd.read_csv(q_path)
        base = pd.read_csv(node_path)
        q_columns = ["node_id", "z", "coordination", "mean_s6", "fraction_s6_above_threshold", *Q_PROPERTIES]
        merged = base.merge(q[q_columns], on="node_id", how="left", validate="one_to_one")
        merged["solid_s6_connections"] = np.rint(
            merged["coordination"] * merged["fraction_s6_above_threshold"].fillna(0)
        ).astype(int)
        merged["crystal_like"] = merged["solid_s6_connections"] >= int(
            cfg.get("crystal_like_min_s6_connections", 7)
        )
        node_frames.append(merged)
        edge_frames.append(pd.read_csv(edge_path))
    if not node_frames:
        raise FileNotFoundError(f"No corrected Job 4 Ruby tables found below {root}")
    return pd.concat(node_frames, ignore_index=True), pd.concat(edge_frames, ignore_index=True)


def load_crystals(cfg):
    root = Path(cfg["output_root"]) / "job4_bond_order"
    frames = []
    for structure in STRUCTURES:
        path = root / f"crystal_{structure}_nodes.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        graph = crystal_graph(
            structure,
            cfg["particle_diameter"],
            cfg["contact_distance_tolerance_fraction"],
        )
        clustering = nx.clustering(graph)
        frame["degree"] = frame["coordination"]
        frame["clustering"] = frame["node_id"].map(clustering)
        frame["structure"] = structure
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def numeric_edges(groups, column, bins, log=False):
    values = np.concatenate([safe_values(frame, column) for _, frame in groups])
    if log:
        values = values[values > 0]
    if not len(values):
        raise ValueError(f"No finite values for {column}")
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.01, 1e-12)
        lo, hi = lo - pad, hi + pad
    if log:
        return np.geomspace(lo, hi, bins + 1)
    if np.allclose(values, np.round(values), rtol=0, atol=1e-10) and hi - lo + 1 <= bins:
        return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    return np.linspace(lo, hi, bins + 1)


@dataclass
class HistogramScale:
    x_edges: np.ndarray
    y_edges: np.ndarray
    vmin: float
    vmax: float
    y_log: bool


def make_scale(groups, x, y, bins, y_log=False):
    x_edges = numeric_edges(groups, x, bins)
    y_edges = numeric_edges(groups, y, bins, log=y_log)
    probabilities = []
    for _, frame in groups:
        xv = pd.to_numeric(frame[x], errors="coerce").to_numpy(float)
        yv = pd.to_numeric(frame[y], errors="coerce").to_numpy(float)
        good = np.isfinite(xv) & np.isfinite(yv) & ((yv > 0) if y_log else True)
        counts = np.histogram2d(xv[good], yv[good], bins=(x_edges, y_edges))[0]
        probabilities.append(counts / counts.sum() if counts.sum() else counts)
    positive = np.concatenate([p[p > 0] for p in probabilities if np.any(p > 0)])
    if not len(positive):
        raise ValueError(f"No populated bins for {x} versus {y}")
    return HistogramScale(x_edges, y_edges, float(positive.min()), float(positive.max()), y_log)


def render_binned(groups, x, y, scale, path, title, ncols=None):
    n = len(groups)
    ncols = ncols or min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad("white")
    image = None
    for ax, (panel, frame) in zip(axes.flat, groups):
        xv = pd.to_numeric(frame[x], errors="coerce").to_numpy(float)
        yv = pd.to_numeric(frame[y], errors="coerce").to_numpy(float)
        good = np.isfinite(xv) & np.isfinite(yv) & ((yv > 0) if scale.y_log else True)
        counts = np.histogram2d(xv[good], yv[good], bins=(scale.x_edges, scale.y_edges))[0]
        probability = counts / counts.sum() if counts.sum() else counts
        image = ax.pcolormesh(
            scale.x_edges, scale.y_edges, np.ma.masked_less_equal(probability.T, 0),
            shading="auto", cmap=cmap, norm=LogNorm(scale.vmin, scale.vmax),
        )
        ax.set(xlim=(scale.x_edges[0], scale.x_edges[-1]),
               ylim=(scale.y_edges[0], scale.y_edges[-1]),
               xlabel=label(x), ylabel=label(y), title=f"{panel} (n={good.sum():,})")
        if scale.y_log:
            ax.set_yscale("log")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.10, top=0.88, wspace=0.28, hspace=0.38)
    if image is not None:
        color_axis = fig.add_axes((0.86, 0.16, 0.025, 0.68))
        fig.colorbar(image, cax=color_axis, label="Fraction per bin (log color)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scale_row(path, x, y, scale):
    return {
        "plot": str(path), "x_property": x, "y_property": y,
        "x_min": scale.x_edges[0], "x_max": scale.x_edges[-1],
        "y_min": scale.y_edges[0], "y_max": scale.y_edges[-1],
        "x_bins": len(scale.x_edges) - 1, "y_bins": len(scale.y_edges) - 1,
        "color_min_positive_fraction": scale.vmin,
        "color_max_fraction": scale.vmax, "y_scale": "log" if scale.y_log else "linear",
        "bin_value": "fraction_of_finite_paired_observations",
    }


def distribution_plots(nodes, cfg, bond_root, bins):
    colors = dict(zip(cfg["angles"], ("#4C72B0", "#C44E52", "#55A868", "#8172B2")))
    simulation_means, pooled_rows, test_rows = [], [], []
    geometry_dir = bond_root / "distributions" / "geometry_comparison"
    force_dir = bond_root / "distributions" / "force_split"
    geometry_box = bond_root / "simulation_mean_boxplots" / "geometry_comparison"
    force_box = bond_root / "simulation_mean_boxplots" / "force_split"
    for directory in (geometry_dir, force_dir, geometry_box, force_box):
        directory.mkdir(parents=True, exist_ok=True)
    for prop in Q_PROPERTIES:
        edge_groups = [(f"{a} {g}", nodes[(nodes.angle == a) & (nodes.group == g)])
                       for a in cfg["angles"] for g in ("non_high_force", "high_force")]
        edges = numeric_edges(edge_groups, prop, bins)

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for angle in cfg["angles"]:
            values = safe_values(nodes[nodes.angle == angle], prop)
            ax.hist(values, bins=edges, density=True, histtype="step", linewidth=2,
                    label=angle, color=colors[angle])
            pooled_rows.append({"property": prop, "comparison": "geometry", "angle": angle,
                                "group": "all", "n": len(values), "mean": np.mean(values),
                                "median": np.median(values), "std": np.std(values, ddof=1)})
        ax.set(xlabel=label(prop), ylabel="Density", title=f"{label(prop)}: geometry comparison",
               xlim=(edges[0], edges[-1]))
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(geometry_dir / f"{prop}.png", dpi=220); plt.close(fig)

        fig, axes = plt.subplots(1, len(cfg["angles"]), figsize=(5.2 * len(cfg["angles"]), 4.2),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, angle in zip(axes.flat, cfg["angles"]):
            for group in ("non_high_force", "high_force"):
                values = safe_values(nodes[(nodes.angle == angle) & (nodes.group == group)], prop)
                if len(values):
                    ax.hist(values, bins=edges, density=True, histtype="step", linewidth=2,
                            label=group.replace("_", " "), color=GROUP_COLORS[group])
                else:
                    ax.plot([], [], linewidth=2, label=group.replace("_", " "),
                            color=GROUP_COLORS[group])
                pooled_rows.append({"property": prop, "comparison": "force_split", "angle": angle,
                                    "group": group, "n": len(values),
                                    "mean": np.mean(values) if len(values) else np.nan,
                                    "median": np.median(values) if len(values) else np.nan,
                                    "std": np.std(values, ddof=1) if len(values) > 1 else np.nan})
            if not len(nodes[(nodes.angle == angle) & (nodes.group == "high_force")]):
                ax.text(.5, .93, "No high-force particles\nunder fixed threshold",
                        transform=ax.transAxes, ha="center", va="top", fontsize=9,
                        color=GROUP_COLORS["high_force"])
            ax.set(title=angle, xlabel=label(prop), ylabel="Density", xlim=(edges[0], edges[-1]))
            ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(force_dir / f"{prop}.png", dpi=220); plt.close(fig)

        means = nodes.groupby(["angle", "sim_idx", "group"], as_index=False)[prop].mean()
        all_means = nodes.groupby(["angle", "sim_idx"], as_index=False)[prop].mean()
        all_means["group"] = "all"
        for row in pd.concat([all_means, means], ignore_index=True).itertuples(index=False):
            simulation_means.append({"property": prop, "angle": row.angle,
                                     "sim_idx": row.sim_idx, "group": row.group,
                                     "simulation_mean": getattr(row, prop)})

        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        arrays = [finite(all_means.loc[all_means.angle == angle, prop]) for angle in cfg["angles"]]
        boxes = ax.boxplot(arrays, labels=cfg["angles"], patch_artist=True)
        for patch, angle in zip(boxes["boxes"], cfg["angles"]): patch.set_facecolor(colors[angle]); patch.set_alpha(.35)
        for i, (angle, values) in enumerate(zip(cfg["angles"], arrays), 1):
            rng = np.random.default_rng(cfg["random_seed"] + i)
            ax.scatter(i + rng.uniform(-.08, .08, len(values)), values, s=18, color=colors[angle], alpha=.75)
        ax.set(ylabel=f"Simulation mean {label(prop)}", title=f"{label(prop)} by geometry")
        fig.tight_layout(); fig.savefig(geometry_box / f"{prop}.png", dpi=220); plt.close(fig)

        categories = [(a, g) for a in cfg["angles"] for g in ("non_high_force", "high_force")]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        arrays = [finite(means.loc[(means.angle == a) & (means.group == g), prop]) for a, g in categories]
        boxes = ax.boxplot(arrays, labels=[f"{a}\n{'non-high' if g == 'non_high_force' else 'high'}" for a, g in categories], patch_artist=True)
        for patch, (_, group) in zip(boxes["boxes"], categories): patch.set_facecolor(GROUP_COLORS[group]); patch.set_alpha(.35)
        for i, ((_, group), values) in enumerate(zip(categories, arrays), 1):
            rng = np.random.default_rng(cfg["random_seed"] + 10 + i)
            ax.scatter(i + rng.uniform(-.08, .08, len(values)), values, s=16, color=GROUP_COLORS[group], alpha=.75)
        ax.set(ylabel=f"Simulation mean {label(prop)}", title=f"{label(prop)}: force-group comparison")
        fig.tight_layout(); fig.savefig(force_box / f"{prop}.png", dpi=220); plt.close(fig)

        if len(cfg["angles"]) == 2:
            a0, a1 = cfg["angles"]
            test_rows.append({"property": prop, "comparison": f"{a1}_vs_{a0}",
                              **replicate_test(all_means.loc[all_means.angle == a0, prop],
                                               all_means.loc[all_means.angle == a1, prop], paired=False)})
        for angle in cfg["angles"]:
            wide = means[means.angle == angle].pivot(index="sim_idx", columns="group", values=prop).dropna()
            if {"non_high_force", "high_force"}.issubset(wide):
                test_rows.append({"property": prop, "comparison": f"high_vs_non_within_{angle}",
                                  **replicate_test(wide.non_high_force, wide.high_force, paired=True)})
    return pd.DataFrame(simulation_means), pd.DataFrame(pooled_rows), pd.DataFrame(test_rows)


def write_readmes(feature_root, bond_root, cfg, crystal_counts, nodes):
    minimum = int(cfg.get("crystal_like_min_s6_connections", 7))
    angle_text = ", ".join(cfg["angles"])
    feature_root.joinpath("README.md").write_text(f"""# Bivariate feature relationships

Two-dimensional probability-bin plots for the configured graph state.

- `relationship_plots/hydrostress_vs_node_properties/`: hydrostatic stress
  versus each eligible non-stress node property for {angle_text}.
- `relationship_plots/normal_force_vs_edge_properties/`: normal force versus
  contact angle and no-wall curvature.
- `relationship_plots/selected_node_pair/`: degree versus clustering for the
  whole, high-force, and non-high-force populations, plus ideal crystals.
- `tables/plot_scales.csv`: exact shared axis, bin, and color limits.

Each panel is normalized by its own number of finite paired observations. All
panels for the same property pair share axes, bin edges, and logarithmic color
limits. Normal force uses logarithmic y bins; all other axes are linear.
""")
    count_text = ", ".join(f"{row.angle}: {row.crystal_like_nodes}" for row in crystal_counts.itertuples())
    crystal_total = int(crystal_counts["crystal_like_nodes"].sum())
    if crystal_total:
        particle_word = "particle" if crystal_total == 1 else "particles"
        crystal_plot_text = (
            f"It selected {crystal_total} Ruby {particle_word} ({count_text}); the "
            "shared-scale q-pair plots are in "
            "`relationship_plots/binned/crystal_like_particles/`."
        )
    else:
        crystal_plot_text = (
            f"It selected no Ruby particles ({count_text}), so no empty or "
            "threshold-weakened Ruby crystal-like plot was produced; ideal crystal "
            "reference plots are provided instead."
        )
    high_force_total = int((nodes["group"] == "high_force").sum())
    if high_force_total:
        force_note = f"The configured force labeling selected {high_force_total:,} high-force particles."
    else:
        force_note = (
            "The fixed final-load force threshold selected no jamming particles. "
            "High-force panels are therefore explicitly marked as empty; the "
            "threshold was not weakened."
        )
    bond_root.joinpath("README.md").write_text(f"""# Bond order

- `distributions/geometry_comparison/`: distributions across {angle_text}.
- `distributions/force_split/`: high-force versus non-high-force distributions
  within each geometry.
- `distributions/*.png`: original Ruby/ideal-crystal q and s6 distributions.
- `simulation_mean_boxplots/`: the corresponding simulation-replicate views.
- `relationship_plots/binned/whole_high_non_high/`: shared-scale binned q pairs.
- `relationship_plots/binned/crystal_like_particles/`: q pairs for particles
  selected by the configured operational crystal-like rule, when any exist.
- `relationship_plots/binned/hydrostress/`: hydrostatic stress versus q.
- `relationship_plots/binned/ideal_crystal_references/`: SC/BCC/FCC/HCP q-pair
  references on the same scales used for Ruby particles.
- `relationship_plots/*.png`: original q scatter plots, PCA, and heatmap.
- `sample_systems/`: representative three-dimensional q fields.
- `tables/bivariate_*`: simulation means, pooled summaries, tests, plot scales,
  and crystal-like counts.
- `tables/`: original node/contact q values, summaries, and crystal distances.
- `crystal_references/` and `artifacts/`: restartable ideal-crystal and
  per-simulation calculations.

The existing contact-order threshold is `s6 > {cfg['s6_threshold']}`. The
configured operational particle-level crystal-like rule used here requires at
least {minimum} such connections. This threshold combination is not presented
as a universal literature standard. {crystal_plot_text}

{force_note}

All distributions use one pooled bin grid per property. Every binned comparison
uses probability mass per bin and identical axes and logarithmic color limits
for all panels of that property pair.
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bins = int(args.bins or cfg.get("bivariate_bins", 50))
    nodes, edges = load_tables(cfg)
    crystals = load_crystals(cfg)
    selected_x, selected_y = cfg.get("selected_node_property_pair", ["degree", "clustering"])
    missing_node = [p for p in (*NODE_PROPERTIES, "stress_hydro", selected_x, selected_y, *Q_PROPERTIES) if p not in nodes]
    missing_edge = [p for p in (*EDGE_PROPERTIES, "normal_force") if p not in edges]
    if missing_node or missing_edge:
        raise KeyError({"missing_node_properties": missing_node, "missing_edge_properties": missing_edge})
    if args.dry_run:
        print({"nodes": len(nodes), "edges": len(edges), "crystal_nodes": len(crystals),
               "node_relationships": len(NODE_PROPERTIES), "edge_relationships": len(EDGE_PROPERTIES),
               "selected_pair": [selected_x, selected_y], "bins": bins})
        return

    # Keep presentation products beside the pipeline's artifact directory for
    # every state.  Deriving this from output_root avoids assuming that
    # existing_results is the parent of 1_network_property_comparison (it is a
    # graph-data directory for the final-load and jamming configurations).
    analysis_root = Path(cfg.get("analysis_root", Path(cfg["output_root"]).parents[1]))
    feature_root = analysis_root / "feature_relationships"
    bond_root = analysis_root / "bond_order"
    feature_tables = feature_root / "tables"
    bond_tables = bond_root / "tables"
    feature_tables.mkdir(parents=True, exist_ok=True); bond_tables.mkdir(parents=True, exist_ok=True)
    feature_scales, bond_scales = [], []

    geometry_groups_nodes = [(angle, nodes[nodes.angle == angle]) for angle in cfg["angles"]]
    geometry_groups_edges = [(angle, edges[edges.angle == angle]) for angle in cfg["angles"]]
    for prop in NODE_PROPERTIES:
        scale = make_scale(geometry_groups_nodes, prop, "stress_hydro", bins)
        path = feature_root / "relationship_plots" / "hydrostress_vs_node_properties" / f"{prop}.png"
        render_binned(geometry_groups_nodes, prop, "stress_hydro", scale, path,
                      f"Hydrostatic stress versus {label(prop).lower()}", ncols=len(cfg["angles"]))
        feature_scales.append(scale_row(path.relative_to(feature_root), prop, "stress_hydro", scale))
    for prop in EDGE_PROPERTIES:
        scale = make_scale(geometry_groups_edges, prop, "normal_force", bins, y_log=True)
        path = feature_root / "relationship_plots" / "normal_force_vs_edge_properties" / f"{prop}.png"
        render_binned(geometry_groups_edges, prop, "normal_force", scale, path,
                      f"Normal force versus {label(prop).lower()}", ncols=len(cfg["angles"]))
        feature_scales.append(scale_row(path.relative_to(feature_root), prop, "normal_force", scale))

    selected_groups = [(f"{angle}: {name.replace('_', ' ')}",
                        nodes[(nodes.angle == angle) & (nodes.group == name)] if name != "whole" else nodes[nodes.angle == angle])
                       for name in ("whole", "high_force", "non_high_force") for angle in cfg["angles"]]
    crystal_like_groups = [(angle, nodes[(nodes.angle == angle) & nodes.crystal_like])
                           for angle in cfg["angles"]]
    crystal_like_groups = [(name, frame) for name, frame in crystal_like_groups if len(frame)]
    crystal_groups = [(structure, crystals[crystals.structure == structure]) for structure in STRUCTURES]
    selected_scale = make_scale(selected_groups + crystal_like_groups + crystal_groups,
                                selected_x, selected_y, bins)
    path = feature_root / "relationship_plots" / "selected_node_pair" / "whole_high_non_high" / f"{selected_x}_vs_{selected_y}.png"
    render_binned(selected_groups, selected_x, selected_y, selected_scale, path,
                  f"{label(selected_y)} versus {label(selected_x).lower()}: Ruby particles", ncols=len(cfg["angles"]))
    feature_scales.append(scale_row(path.relative_to(feature_root), selected_x, selected_y, selected_scale))
    if crystal_like_groups:
        path = feature_root / "relationship_plots" / "selected_node_pair" / "crystal_like_particles" / f"{selected_x}_vs_{selected_y}.png"
        render_binned(crystal_like_groups, selected_x, selected_y, selected_scale, path,
                      f"{label(selected_y)} versus {label(selected_x).lower()}: operational crystal-like particles",
                      ncols=len(cfg["angles"]))
        feature_scales.append(scale_row(path.relative_to(feature_root), selected_x, selected_y, selected_scale))
    path = feature_root / "relationship_plots" / "selected_node_pair" / "ideal_crystal_references" / f"{selected_x}_vs_{selected_y}.png"
    render_binned(crystal_groups, selected_x, selected_y, selected_scale, path,
                  f"{label(selected_y)} versus {label(selected_x).lower()}: ideal crystals", ncols=2)
    feature_scales.append(scale_row(path.relative_to(feature_root), selected_x, selected_y, selected_scale))

    q_group_sets = {
        "whole_high_non_high": selected_groups,
        "ideal_crystal_references": crystal_groups,
    }
    for x, y in (("q4", "q6"), ("qbar4", "qbar6")):
        scale = make_scale(selected_groups + crystal_like_groups + crystal_groups, x, y, bins)
        for folder, groups in q_group_sets.items():
            path = bond_root / "relationship_plots" / "binned" / folder / f"{x}_vs_{y}.png"
            comparison_title = ("whole, high-force, and non-high-force Ruby particles"
                                if folder == "whole_high_non_high" else "ideal crystal references")
            render_binned(groups, x, y, scale, path,
                          f"{label(y)} versus {label(x)}: {comparison_title}",
                          ncols=len(cfg["angles"]) if folder == "whole_high_non_high" else 2)
            bond_scales.append(scale_row(path.relative_to(bond_root), x, y, scale))
        if crystal_like_groups:
            path = bond_root / "relationship_plots" / "binned" / "crystal_like_particles" / f"{x}_vs_{y}.png"
            render_binned(crystal_like_groups, x, y, scale, path,
                          f"{label(y)} versus {label(x)}: operational crystal-like particles",
                          ncols=len(cfg["angles"]))
            bond_scales.append(scale_row(path.relative_to(bond_root), x, y, scale))
    for prop in Q_PROPERTIES:
        scale = make_scale(geometry_groups_nodes, prop, "stress_hydro", bins)
        path = bond_root / "relationship_plots" / "binned" / "hydrostress" / f"{prop}.png"
        render_binned(geometry_groups_nodes, prop, "stress_hydro", scale, path,
                      f"Hydrostatic stress versus {label(prop)}", ncols=len(cfg["angles"]))
        bond_scales.append(scale_row(path.relative_to(bond_root), prop, "stress_hydro", scale))

    means, pooled, tests = distribution_plots(nodes, cfg, bond_root, bins)
    crystal_counts = nodes.groupby("angle", as_index=False).agg(
        total_nodes=("node_id", "size"), crystal_like_nodes=("crystal_like", "sum"),
        maximum_s6_connections=("solid_s6_connections", "max"),
    )
    crystal_counts["criterion_minimum_s6_connections"] = int(cfg.get("crystal_like_min_s6_connections", 7))
    crystal_counts["s6_threshold"] = cfg["s6_threshold"]
    atomic_csv(feature_tables / "plot_scales.csv", pd.DataFrame(feature_scales))
    atomic_csv(bond_tables / "bivariate_plot_scales.csv", pd.DataFrame(bond_scales))
    atomic_csv(bond_tables / "bivariate_simulation_means.csv", means)
    atomic_csv(bond_tables / "bivariate_pooled_summaries.csv", pooled)
    atomic_csv(bond_tables / "bivariate_simulation_tests.csv", tests)
    atomic_csv(bond_tables / "crystal_like_particle_counts.csv", crystal_counts)
    atomic_json(feature_tables / "analysis_metadata.json", {
        "config": str(Path(args.config).resolve()), "bins_requested": bins,
        "node_rows": len(nodes), "edge_rows": len(edges), "crystal_rows": len(crystals),
        "node_properties": NODE_PROPERTIES, "edge_properties": EDGE_PROPERTIES,
        "selected_node_pair": [selected_x, selected_y],
        "normalization": "fraction of finite paired observations per panel",
        "shared_scale_rule": "same property pair shares axes, bins, and color limits",
    })
    write_readmes(feature_root, bond_root, cfg, crystal_counts, nodes)
    new_bond_plots = (len(list((bond_root / "relationship_plots" / "binned").rglob("*.png")))
                      + len(list((bond_root / "distributions" / "geometry_comparison").rglob("*.png")))
                      + len(list((bond_root / "distributions" / "force_split").rglob("*.png")))
                      + len(list((bond_root / "simulation_mean_boxplots").rglob("*.png"))))
    print({"feature_plots": len(list(feature_root.rglob("*.png"))),
           "new_bond_plots": new_bond_plots,
           "feature_root": str(feature_root), "bond_root": str(bond_root)})


if __name__ == "__main__":
    main()
