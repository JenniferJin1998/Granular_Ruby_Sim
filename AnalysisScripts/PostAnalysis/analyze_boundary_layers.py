#!/usr/bin/env python3
"""3-D property maps and boundary-shell distribution comparisons.

The analysis works directly from node_features.csv and edge_features.csv.  A
particle is in shell 0 when it participates in an edge tagged
``is_wall_contact``.  Breadth-first search over particle-particle contacts then
assigns shells 1, 2, 3, ... .  A particle-particle edge inherits the smaller
shell of its endpoints. Wall records seed the boundary and are then excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import deque
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]
DATASETS = {
    "PeriodicBoudaries_2026-08-03": PROJECT / "AnalysisResults" / "PeriodicBoudaries" / "2026-08-03" / "GraphPipeline",
    "FinalLoadState": PROJECT / "AnalysisResults" / "FinalLoadState" / "FullGraph_2mean_ref_geom",
}
NODE_EXCLUDE = {"geometry", "sim_idx", "node_id", "x", "y", "z", "is_wall", "in_center_region", "principal_eigenvector", "force_chain_role"}
EDGE_EXCLUDE = {"geometry", "sim_idx", "node1", "node2", "contact_x", "contact_y", "contact_z", "n_x", "n_y", "n_z", "t_x", "t_y", "t_z", "is_core_edge", "is_wall_contact"}
SHELL_ORDER = ["0 (boundary)", "1", "rest (>=2/unreached)"]
VIEWS = [
    ("Perspective", 24, -52),
    ("Along x", 0, 0),
    ("Along y", 0, 90),
    ("Along z", 90, -90),
]
CONTRASTS = [
    ("0 vs rest", [
        ("distance 0", lambda d: d == 0),
        ("distance >=1 or unreachable", lambda d: (d > 0) | ~np.isfinite(d)),
    ]),
    ("0, 1, and rest", [
        ("distance 0", lambda d: d == 0),
        ("distance 1", lambda d: d == 1),
        ("distance >=2 or unreachable", lambda d: (d > 1) | ~np.isfinite(d)),
    ]),
]
BOUNDARY_VIEWS = [
    ("0_vs_rest", "0 versus rest", [
        ("distance 0", lambda d: d == 0),
        ("rest (>=1/unreachable)", lambda d: (d > 0) | ~np.isfinite(d)),
    ]),
    ("0_1_rest", "0, 1, and rest", [
        ("distance 0", lambda d: d == 0),
        ("distance 1", lambda d: d == 1),
        ("rest (>=2/unreachable)", lambda d: (d > 1) | ~np.isfinite(d)),
    ]),
]
GRAY = "#d0d0d0"
LIGHT_RED = "#ef9a9a"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def scalar_properties(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    result = []
    for column in df.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            result.append(column)
    return result


def normalize_endpoint(value):
    """Match CSV numeric strings to integer node IDs while retaining wall labels."""
    if isinstance(value, str):
        stripped = value.strip()
        try:
            number = float(stripped)
            if number.is_integer():
                return int(number)
        except ValueError:
            return stripped
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    return value


def shell_label(distance: float) -> str:
    if not np.isfinite(distance) or distance >= 2:
        return SHELL_ORDER[-1]
    return "0 (boundary)" if int(distance) == 0 else str(int(distance))


def assign_shells(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    node_parts, edge_parts, audit = [], [], []
    keys = sorted(set(map(tuple, nodes[["geometry", "sim_idx"]].drop_duplicates().to_numpy())))
    for geometry, sim_idx in keys:
        n = nodes[(nodes.geometry == geometry) & (nodes.sim_idx == sim_idx)].copy()
        e = edges[(edges.geometry == geometry) & (edges.sim_idx == sim_idx)].copy()
        particle_ids = set(n.node_id.tolist())
        adjacency = {node: [] for node in particle_ids}
        boundary = set()
        for row in e[["node1", "node2", "is_wall_contact"]].itertuples(index=False):
            u, v, wall = row
            u_particle, v_particle = u in particle_ids, v in particle_ids
            if bool(wall):
                if u_particle:
                    boundary.add(u)
                if v_particle:
                    boundary.add(v)
            elif u_particle and v_particle:
                adjacency[u].append(v)
                adjacency[v].append(u)
        distance = {node: math.inf for node in particle_ids}
        queue = deque(boundary)
        for node in boundary:
            distance[node] = 0
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if not math.isfinite(distance[neighbor]):
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        n["boundary_distance"] = n.node_id.map(distance)
        n["boundary_shell"] = pd.Categorical(n.boundary_distance.map(shell_label), SHELL_ORDER, ordered=True)
        # Wall edges are used only to seed the boundary and are then excluded:
        # every analyzed edge is an actual particle-particle contact.
        e = e[(~e.is_wall_contact.astype(bool)) & e.node1.isin(particle_ids) & e.node2.isin(particle_ids)].copy()
        edge_distance = []
        for row in e[["node1", "node2"]].itertuples(index=False):
            edge_distance.append(min(distance.get(row.node1, math.inf), distance.get(row.node2, math.inf)))
        e["boundary_distance"] = edge_distance
        e["boundary_shell"] = pd.Categorical(e.boundary_distance.map(shell_label), SHELL_ORDER, ordered=True)
        counts = n.boundary_shell.value_counts().reindex(SHELL_ORDER, fill_value=0)
        audit.append({"geometry": geometry, "sim_idx": int(sim_idx), "particle_nodes": len(n), "particle_contacts": len(e),
                      "boundary_nodes": len(boundary), "unreached_nodes": int(np.isinf(n.boundary_distance).sum()),
                      **{f"node_shell_{safe_name(k)}": int(v) for k, v in counts.items()}})
        node_parts.append(n)
        edge_parts.append(e)
    return pd.concat(node_parts, ignore_index=True), pd.concat(edge_parts, ignore_index=True), audit


def color_limits(df: pd.DataFrame, properties: list[str], lower: float, upper: float) -> pd.DataFrame:
    rows = []
    for prop in properties:
        x = pd.to_numeric(df[prop], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        if x.size:
            lo, hi = np.percentile(x, [lower, upper])
            if hi <= lo:
                hi = lo + max(abs(lo) * 1e-9, 1e-12)
            rows.append({"property": prop, "cmin": lo, "cmax": hi, "lower_percentile": lower, "upper_percentile": upper})
    return pd.DataFrame(rows)


def representative_simulations(nodes: pd.DataFrame) -> dict[str, int]:
    result = {}
    for geometry, group in nodes.groupby("geometry", sort=True):
        sims = sorted(group.sim_idx.unique())
        result[str(geometry)] = int(sims[len(sims) // 2])
    return result


def minimum_image(delta: np.ndarray, box_lengths: dict[int, float], periodic_axes: list[int]) -> np.ndarray:
    result = np.asarray(delta, float).copy()
    for axis in periodic_axes:
        length = float(box_lengths[axis])
        result[axis] -= length * np.round(result[axis] / length)
    return result


def split_periodic_segment(start: np.ndarray, delta: np.ndarray, box_lengths: dict[int, float], periodic_axes: list[int]) -> list[np.ndarray]:
    """Split a minimum-image contact where it crosses an orthogonal periodic face."""
    current = np.asarray(start, float).copy()
    remaining = np.asarray(delta, float).copy()
    pieces = []
    for _ in range(len(periodic_axes) + 1):
        endpoint = current + remaining
        crossings = []
        for axis in periodic_axes:
            length = float(box_lengths[axis])
            if endpoint[axis] < 0 and remaining[axis] < 0:
                crossings.append(((0 - current[axis]) / remaining[axis], axis, length))
            elif endpoint[axis] > length and remaining[axis] > 0:
                crossings.append(((length - current[axis]) / remaining[axis], axis, 0.0))
        crossings = [item for item in crossings if 0 < item[0] < 1]
        if not crossings:
            pieces.append(np.vstack((current, endpoint)))
            break
        crossing_t = min(item[0] for item in crossings)
        hit = current + crossing_t * remaining
        pieces.append(np.vstack((current, hit)))
        remaining = (1 - crossing_t) * remaining
        current = hit.copy()
        for item_t, axis, wrapped in crossings:
            if np.isclose(item_t, crossing_t):
                current[axis] = wrapped
    return pieces


def contact_segments(core: pd.DataFrame, pos: dict, box_lengths: dict[int, float], periodic_axes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return drawable segments and the source-edge row index for each piece."""
    segments, source_indices = [], []
    for index, (u, v) in enumerate(core[["node1", "node2"]].itertuples(index=False, name=None)):
        start = pos[u]
        raw_delta = pos[v] - start
        if periodic_axes:
            delta = minimum_image(raw_delta, box_lengths, periodic_axes)
            pieces = split_periodic_segment(start, delta, box_lengths, periodic_axes)
        else:
            pieces = [np.vstack((start, pos[v]))]
        segments.extend(pieces)
        source_indices.extend([index] * len(pieces))
    return np.asarray(segments), np.asarray(source_indices, dtype=int)


def plot_3d_maps(nodes: pd.DataFrame, edges: pd.DataFrame, properties: list[str], level: str,
                 limits: pd.DataFrame, output: Path, representative: dict[str, int],
                 box_lengths=None, periodic_axes=None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    limit_map = limits.set_index("property")[["cmin", "cmax"]].to_dict("index")
    lower_pct = float(limits.lower_percentile.iloc[0])
    upper_pct = float(limits.upper_percentile.iloc[0])
    geometries = list(representative)
    box_lengths = box_lengths or {}
    periodic_axes = periodic_axes or []
    for prop in properties:
        fig = plt.figure(figsize=(4.6 * len(geometries), 3.8 * len(VIEWS)))
        lo, hi = limit_map[prop]["cmin"], limit_map[prop]["cmax"]
        norm = colors.Normalize(lo, hi, clip=True)
        cmap = plt.get_cmap("viridis")
        plotted = False
        used_axes = []
        for column, geometry in enumerate(geometries):
            sim_idx = representative[geometry]
            n = nodes[(nodes.geometry == geometry) & (nodes.sim_idx == sim_idx)].copy()
            e = edges[(edges.geometry == geometry) & (edges.sim_idx == sim_idx)].copy()
            pos = {row.node_id: np.asarray([row.x, row.y, row.z], float) for row in n.itertuples()}
            core = e[e.node1.isin(pos) & e.node2.isin(pos)].copy()
            segs, segment_sources = contact_segments(core, pos, box_lengths, periodic_axes)
            xyz = n[["x", "y", "z"]].to_numpy(float)
            force_nodes = n["is_force_chain_node"].fillna(False).astype(bool).to_numpy() if "is_force_chain_node" in n else np.zeros(len(n), bool)
            force_edges = core["is_high_force"].fillna(False).astype(bool).to_numpy() if "is_high_force" in core else np.zeros(len(core), bool)
            force_segments = force_edges[segment_sources] if len(segment_sources) else np.zeros(0, bool)
            all_xyz = n[["x", "y", "z"]].to_numpy(float)
            center = (all_xyz.min(0) + all_xyz.max(0)) / 2
            radius = max(np.ptp(all_xyz, axis=0)) / 2 * 1.04
            for row, (view_name, elev, azim) in enumerate(VIEWS):
                index = row * len(geometries) + column + 1
                ax = fig.add_subplot(len(VIEWS), len(geometries), index, projection="3d")
                used_axes.append(ax)
                if level == "node":
                    if len(segs):
                        ax.add_collection3d(Line3DCollection(segs, colors=GRAY, linewidths=.28, alpha=.25, rasterized=True))
                        if force_segments.any():
                            ax.add_collection3d(Line3DCollection(segs[force_segments], colors=LIGHT_RED, linewidths=.85, alpha=.72, rasterized=True))
                    vals = pd.to_numeric(n[prop], errors="coerce").to_numpy(float)
                    valid = np.isfinite(vals) & force_nodes
                    ax.scatter(*xyz.T, c=GRAY, s=6, linewidths=0, alpha=.42, rasterized=True)
                    if valid.any():
                        ax.scatter(*xyz[valid].T, c=vals[valid], cmap=cmap, norm=norm, s=14, linewidths=0, alpha=.98, rasterized=True)
                    plotted |= bool(valid.any())
                else:
                    ax.scatter(*xyz.T, c=GRAY, s=6, linewidths=0, alpha=.42, rasterized=True)
                    if force_nodes.any():
                        ax.scatter(*xyz[force_nodes].T, c=LIGHT_RED, s=12, linewidths=0, alpha=.82, rasterized=True)
                    vals = pd.to_numeric(core[prop], errors="coerce").to_numpy(float)
                    segment_vals = vals[segment_sources] if len(segment_sources) else np.asarray([])
                    valid = np.isfinite(segment_vals) & force_segments
                    if len(segs):
                        ax.add_collection3d(Line3DCollection(segs, colors=GRAY, linewidths=.28, alpha=.25, rasterized=True))
                    if valid.any():
                        ax.add_collection3d(Line3DCollection(segs[valid], colors=cmap(norm(segment_vals[valid])), linewidths=1.15, alpha=.98, rasterized=True))
                        plotted = True
                mins, maxs = all_xyz.min(0), all_xyz.max(0)
                span = np.maximum(maxs - mins, np.finfo(float).eps)
                ax.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]), zlim=(mins[2], maxs[2]))
                try:
                    ax.set_box_aspect(span, zoom=1.55)
                except TypeError:
                    ax.set_box_aspect(span)
                    ax.dist = 6.5
                ax.set_axis_off(); ax.view_init(elev, azim)
                ax.set_proj_type("persp" if view_name == "Perspective" else "ortho")
                title = f"{geometry}, simulation {sim_idx}" if row == 0 else view_name
                ax.set_title(title, fontsize=12, pad=1)
                if column == 0:
                    ax.text2D(-.04, .50, view_name, transform=ax.transAxes, rotation=90,
                              ha="center", va="center", fontsize=12, fontweight="bold")
        if plotted:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            color_axis = fig.add_axes([.968, .30, .012, .40])
            fig.colorbar(sm, cax=color_axis, label=f"{prop} (range {lo:.3g} to {hi:.3g})")
        fig.suptitle(f"{level.title()} property: {prop}\ncommon range across all geometries: P{lower_pct:g}–P{upper_pct:g}", fontsize=16)
        fig.subplots_adjust(left=.025, right=.955, bottom=.015, top=.93, wspace=.01, hspace=.02)
        fig.savefig(output / f"{safe_name(prop)}.png", dpi=150)
        plt.close(fig)


def distribution_outputs(df: pd.DataFrame, properties: list[str], level: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_dir = output / f"{level}_property_distributions"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, test_rows = [], []
    geometries = sorted(df.geometry.unique())
    for prop in properties:
        fig, axes = plt.subplots(len(CONTRASTS), len(geometries), figsize=(5 * len(geometries), 8), squeeze=False, constrained_layout=True)
        pooled = pd.to_numeric(df[prop], errors="coerce").to_numpy(float)
        pooled = pooled[np.isfinite(pooled)]
        lo, hi = np.percentile(pooled, [1, 99]) if pooled.size else (0, 1)
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 51)
        for column, geometry in enumerate(geometries):
            gd = df[df.geometry == geometry]
            for shell in SHELL_ORDER:
                x = pd.to_numeric(gd.loc[gd.boundary_shell == shell, prop], errors="coerce").to_numpy(float)
                x = x[np.isfinite(x)]
                if x.size:
                    clipped = x[(x >= lo) & (x <= hi)]
                summary_rows.append({"entity": level, "property": prop, "geometry": geometry, "boundary_shell": shell,
                                     "n": int(x.size), "mean": np.mean(x) if x.size else np.nan,
                                     "median": np.median(x) if x.size else np.nan, "std": np.std(x, ddof=1) if x.size > 1 else np.nan,
                                     "p05": np.percentile(x, 5) if x.size else np.nan, "p95": np.percentile(x, 95) if x.size else np.nan})
            distances = gd.boundary_distance.to_numpy(float)
            for row, (contrast, group_specs) in enumerate(CONTRASTS):
                ax = axes[row, column]
                groups = []
                palette = ["#6A3D9A", "#377EB8", "#1B9E77"]
                for (label, selector), color in zip(group_specs, palette):
                    mask = selector(distances)
                    values = pd.to_numeric(gd.loc[mask, prop], errors="coerce").dropna().to_numpy(float)
                    groups.append((label, values))
                    clipped = values[(values >= lo) & (values <= hi)]
                    if clipped.size:
                        ax.hist(clipped, bins=bins, density=True, histtype="step", linewidth=1.55,
                                color=color, label=f"{label} (n={values.size:,})")
                ax.set(title=f"{geometry}: {contrast}", xlabel=prop, ylabel="density", xlim=(lo, hi))
                ax.grid(alpha=.18); ax.legend(fontsize=8, frameon=False)
                for first in range(len(groups)):
                    for second in range(first + 1, len(groups)):
                        label_a, xa = groups[first]
                        label_b, xb = groups[second]
                        if xa.size and xb.size:
                            ks = stats.ks_2samp(xa, xb)
                            test_rows.append({"entity": level, "property": prop, "geometry": geometry, "contrast": contrast,
                                              "group_a": label_a, "group_b": label_b, "n_a": xa.size, "n_b": xb.size,
                                              "ks_statistic": ks.statistic, "ks_pvalue": ks.pvalue,
                                              "wasserstein_distance": stats.wasserstein_distance(xa, xb)})
        fig.suptitle(f"{level.title()} {prop}: requested boundary-distance contrasts")
        fig.savefig(plot_dir / f"{safe_name(prop)}.png", dpi=170)
        plt.close(fig)
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def benjamini_hochberg(pvalues: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not valid.size:
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * valid.size / np.arange(1, valid.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def simulation_summary_outputs(df: pd.DataFrame, properties: list[str], level: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare per-simulation means/medians across geometry angles."""
    plot_dir = output / f"{level}_simulation_mean_median"
    plot_dir.mkdir(parents=True, exist_ok=True)
    geometries = sorted(df.geometry.unique())
    value_rows, test_rows = [], []
    rng = np.random.default_rng(20260901)
    point_colors = plt.get_cmap("tab10")(np.linspace(0, .8, max(len(geometries), 2)))
    for prop in properties:
        for (geometry, sim_idx, shell), group in df.groupby(["geometry", "sim_idx", "boundary_shell"], observed=True, sort=True):
            x = pd.to_numeric(group[prop], errors="coerce").dropna().to_numpy(float)
            if x.size:
                value_rows.append({"entity": level, "property": prop, "geometry": geometry, "sim_idx": int(sim_idx),
                                   "boundary_shell": str(shell), "n_entities": x.size,
                                   "mean": float(np.mean(x)), "median": float(np.median(x))})
        prop_values = pd.DataFrame([row for row in value_rows if row["property"] == prop])
        fig, axes = plt.subplots(2, len(SHELL_ORDER), figsize=(5.2 * len(SHELL_ORDER), 8.5), sharey="row", squeeze=False, constrained_layout=True)
        for row, statistic in enumerate(["mean", "median"]):
            for column, shell in enumerate(SHELL_ORDER):
                ax = axes[row, column]
                arrays = []
                for geometry in geometries:
                    values = prop_values.loc[(prop_values.geometry == geometry) & (prop_values.boundary_shell == shell), statistic].to_numpy(float)
                    arrays.append(values)
                nonempty = [(i + 1, values) for i, values in enumerate(arrays) if values.size]
                if nonempty:
                    positions, data = zip(*nonempty)
                    bp = ax.boxplot(data, positions=positions, widths=.55, patch_artist=True, showfliers=False,
                                    medianprops={"color": "black", "linewidth": 1.3})
                    for patch, position in zip(bp["boxes"], positions):
                        patch.set_facecolor(point_colors[position - 1]); patch.set_alpha(.25)
                    for position, values in nonempty:
                        jitter = rng.uniform(-.13, .13, size=len(values))
                        ax.scatter(position + jitter, values, s=19, color=point_colors[position - 1], alpha=.80, edgecolors="none")
                ax.set_xticks(range(1, len(geometries) + 1))
                ax.set_xticklabels(geometries)
                ax.set(title=shell, xlabel="boundary angle", ylabel=f"per-simulation {statistic}" if column == 0 else "")
                ax.grid(axis="y", alpha=.20)
                for geom_a, geom_b in combinations(geometries, 2):
                    xa = prop_values.loc[(prop_values.geometry == geom_a) & (prop_values.boundary_shell == shell), statistic].to_numpy(float)
                    xb = prop_values.loc[(prop_values.geometry == geom_b) & (prop_values.boundary_shell == shell), statistic].to_numpy(float)
                    if xa.size and xb.size:
                        if np.ptp(np.concatenate([xa, xb])) == 0:
                            u_statistic, pvalue = xa.size * xb.size / 2.0, 1.0
                        else:
                            test = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                            u_statistic, pvalue = test.statistic, test.pvalue
                        test_rows.append({"entity": level, "property": prop, "boundary_shell": shell, "statistic": statistic,
                                          "geometry_a": geom_a, "geometry_b": geom_b, "n_simulations_a": xa.size,
                                          "n_simulations_b": xb.size, "mann_whitney_u": u_statistic, "pvalue": pvalue,
                                          "median_a": np.median(xa), "median_b": np.median(xb)})
        fig.suptitle(f"{level.title()} {prop}: simulation-level summaries across boundary angles", fontsize=14)
        fig.savefig(plot_dir / f"{safe_name(prop)}.png", dpi=170)
        plt.close(fig)
    values = pd.DataFrame(value_rows)
    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["pvalue_bh_all_tests"] = benjamini_hochberg(tests.pvalue)
    return values, tests


def force_label(level: str) -> str:
    return "is_force_chain_node" if level == "node" else "is_high_force"


def force_split_histogram_outputs(df: pd.DataFrame, properties: list[str], level: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overlay stored force/non-force populations within requested boundary groups."""
    plot_dir = output / f"{level}_force_split_histograms"
    plot_dir.mkdir(parents=True, exist_ok=True)
    geometries = sorted(df.geometry.unique())
    force_mask_all = df[force_label(level)].fillna(False).astype(bool).to_numpy()
    summary_rows, test_rows = [], []
    for prop in properties:
        pooled = pd.to_numeric(df[prop], errors="coerce").to_numpy(float)
        pooled = pooled[np.isfinite(pooled)]
        lo, hi = np.percentile(pooled, [1, 99]) if pooled.size else (0, 1)
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 51)
        for view_key, view_title, categories in BOUNDARY_VIEWS:
            fig, axes = plt.subplots(len(categories), len(geometries), figsize=(5 * len(geometries), 3.5 * len(categories)),
                                     squeeze=False, constrained_layout=True)
            for column, geometry in enumerate(geometries):
                geom_mask = (df.geometry.to_numpy() == geometry)
                distances = df.boundary_distance.to_numpy(float)
                for row, (category, selector) in enumerate(categories):
                    ax = axes[row, column]
                    category_mask = geom_mask & selector(distances)
                    groups = []
                    for force_value, force_name, color in [(True, "force cluster", "#D73027"), (False, "non-force", "#4575B4")]:
                        mask = category_mask & (force_mask_all == force_value)
                        values = pd.to_numeric(df.loc[mask, prop], errors="coerce").dropna().to_numpy(float)
                        groups.append((force_name, values))
                        clipped = values[(values >= lo) & (values <= hi)]
                        if clipped.size:
                            ax.hist(clipped, bins=bins, density=True, histtype="step", linewidth=1.55,
                                    color=color, label=f"{force_name} (n={values.size:,})")
                        summary_rows.append({"entity": level, "property": prop, "boundary_view": view_key,
                                             "boundary_group": category, "geometry": geometry, "force_group": force_name,
                                             "n": values.size, "mean": np.mean(values) if values.size else np.nan,
                                             "median": np.median(values) if values.size else np.nan,
                                             "std": np.std(values, ddof=1) if values.size > 1 else np.nan})
                    xa, xb = groups[0][1], groups[1][1]
                    if xa.size and xb.size:
                        ks = stats.ks_2samp(xa, xb)
                        test_rows.append({"entity": level, "property": prop, "boundary_view": view_key,
                                          "boundary_group": category, "geometry": geometry, "n_force": xa.size,
                                          "n_non_force": xb.size, "ks_statistic": ks.statistic, "ks_pvalue": ks.pvalue,
                                          "wasserstein_distance": stats.wasserstein_distance(xa, xb)})
                    ax.set(title=f"{geometry}: {category}", xlabel=prop, ylabel="density", xlim=(lo, hi))
                    ax.grid(alpha=.18); ax.legend(fontsize=8, frameon=False)
            fig.suptitle(f"{level.title()} {prop}: force cluster versus non-force; {view_title}", fontsize=14)
            fig.savefig(plot_dir / f"{safe_name(prop)}_{view_key}.png", dpi=170)
            plt.close(fig)
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def paired_force_test(xa: np.ndarray, xb: np.ndarray) -> tuple[float, float]:
    delta = xa - xb
    if np.ptp(delta) == 0 and delta[0] == 0:
        return 0.0, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = stats.wilcoxon(xa, xb, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def force_split_simulation_outputs(df: pd.DataFrame, properties: list[str], level: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulation-replicate mean/median plots and tests, split by force label."""
    plot_dir = output / f"{level}_force_split_simulation_mean_median"
    plot_dir.mkdir(parents=True, exist_ok=True)
    geometries = sorted(df.geometry.unique())
    force_column = force_label(level)
    value_rows, test_rows = [], []
    rng = np.random.default_rng(20260902)
    force_colors = {"force cluster": "#D73027", "non-force": "#4575B4"}
    for prop in properties:
        for view_key, view_title, categories in BOUNDARY_VIEWS:
            prop_rows = []
            for (geometry, sim_idx), group in df.groupby(["geometry", "sim_idx"], sort=True):
                distances = group.boundary_distance.to_numpy(float)
                forces = group[force_column].fillna(False).astype(bool).to_numpy()
                for category, selector in categories:
                    category_mask = selector(distances)
                    for force_value, force_name in [(True, "force cluster"), (False, "non-force")]:
                        values = pd.to_numeric(group.loc[category_mask & (forces == force_value), prop], errors="coerce").dropna().to_numpy(float)
                        if values.size:
                            record = {"entity": level, "property": prop, "boundary_view": view_key,
                                      "boundary_group": category, "geometry": geometry, "sim_idx": int(sim_idx),
                                      "force_group": force_name, "n_entities": values.size,
                                      "mean": float(np.mean(values)), "median": float(np.median(values))}
                            prop_rows.append(record); value_rows.append(record)
            values_df = pd.DataFrame(prop_rows)
            fig, axes = plt.subplots(2, len(categories), figsize=(5.3 * len(categories), 8.5), sharey="row",
                                     squeeze=False, constrained_layout=True)
            for row, statistic_name in enumerate(["mean", "median"]):
                for column, (category, _) in enumerate(categories):
                    ax = axes[row, column]
                    for geom_index, geometry in enumerate(geometries, 1):
                        for force_name, offset in [("force cluster", -.18), ("non-force", .18)]:
                            subset = values_df[(values_df.geometry == geometry) & (values_df.boundary_group == category) &
                                               (values_df.force_group == force_name)]
                            values = subset[statistic_name].to_numpy(float)
                            if not values.size:
                                continue
                            bp = ax.boxplot([values], positions=[geom_index + offset], widths=.30, patch_artist=True,
                                            showfliers=False, medianprops={"color": "black", "linewidth": 1.2})
                            bp["boxes"][0].set_facecolor(force_colors[force_name]); bp["boxes"][0].set_alpha(.25)
                            jitter = rng.uniform(-.07, .07, size=len(values))
                            ax.scatter(geom_index + offset + jitter, values, s=18, color=force_colors[force_name],
                                       alpha=.78, edgecolors="none")
                    ax.set_xticks(range(1, len(geometries) + 1)); ax.set_xticklabels(geometries)
                    ax.set(title=category, xlabel="boundary angle",
                           ylabel=f"per-simulation {statistic_name}" if column == 0 else "")
                    ax.grid(axis="y", alpha=.20)
                    # Angle comparisons are done separately for each force population.
                    for force_name in force_colors:
                        for geom_a, geom_b in combinations(geometries, 2):
                            xa = values_df.loc[(values_df.geometry == geom_a) & (values_df.boundary_group == category) &
                                               (values_df.force_group == force_name), statistic_name].to_numpy(float)
                            xb = values_df.loc[(values_df.geometry == geom_b) & (values_df.boundary_group == category) &
                                               (values_df.force_group == force_name), statistic_name].to_numpy(float)
                            if xa.size and xb.size:
                                if np.ptp(np.concatenate([xa, xb])) == 0:
                                    u_stat, pvalue = xa.size * xb.size / 2.0, 1.0
                                else:
                                    result = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                                    u_stat, pvalue = result.statistic, result.pvalue
                                test_rows.append({"entity": level, "property": prop, "boundary_view": view_key,
                                                  "boundary_group": category, "statistic": statistic_name,
                                                  "test_type": "angle_mann_whitney", "force_group": force_name,
                                                  "group_a": geom_a, "group_b": geom_b, "n_a": xa.size, "n_b": xb.size,
                                                  "test_statistic": u_stat, "pvalue": pvalue})
                    # Force/non-force comparison is paired by simulation within each angle.
                    for geometry in geometries:
                        paired = values_df[(values_df.geometry == geometry) & (values_df.boundary_group == category)].pivot(
                            index="sim_idx", columns="force_group", values=statistic_name).dropna()
                        if {"force cluster", "non-force"}.issubset(paired.columns) and len(paired):
                            test_stat, pvalue = paired_force_test(paired["force cluster"].to_numpy(float), paired["non-force"].to_numpy(float))
                            test_rows.append({"entity": level, "property": prop, "boundary_view": view_key,
                                              "boundary_group": category, "statistic": statistic_name,
                                              "test_type": "paired_force_vs_nonforce_wilcoxon", "force_group": "paired",
                                              "group_a": f"{geometry}: force cluster", "group_b": f"{geometry}: non-force",
                                              "n_a": len(paired), "n_b": len(paired), "test_statistic": test_stat, "pvalue": pvalue})
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=label) for label, color in force_colors.items()]
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
            fig.suptitle(f"{level.title()} {prop}: force/non-force simulation summaries; {view_title}", fontsize=14)
            fig.savefig(plot_dir / f"{safe_name(prop)}_{view_key}.png", dpi=170)
            plt.close(fig)
    values = pd.DataFrame(value_rows)
    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["pvalue_bh_all_tests"] = benjamini_hochberg(tests.pvalue)
    return values, tests


def write_readme(output: Path, name: str, source: Path, node_props: list[str], edge_props: list[str], representative: dict[str, int]) -> None:
    audit_inventory = ""
    if name.startswith("Periodic"):
        audit_inventory = """| `RAW_EDGE_PROVENANCE_AUDIT.md` | Trace of long-coordinate periodic edges back to the available raw force-contact data. |
| `raw_edge_provenance_audit.csv` | Per-simulation raw-pair versus graph-pair equality counts for 0°. |
"""
    text = f"""# {name}: property and boundary-layer analysis

## What is in this folder

This is a generated, restartable analysis of `{source}`. The source feature tables are not modified.

| Path | Contents |
|---|---|
| `node_properties_3d/` | Force-cluster montage per node property, with perspective and x/y/z projections. Force nodes carry the property color, force edges are light red, and the background network is light gray. |
| `edge_properties_3d/` | Force-cluster montage per edge property. Force edges carry the property color, force nodes are light red, and the background network is light gray. |
| `boundary_distributions/node_property_distributions/` | Two requested views for every node property: 0 vs rest, then three separate curves for 0, 1, and rest. |
| `boundary_distributions/edge_property_distributions/` | The same two contrasts for every particle-contact property. |
| `boundary_distributions/node_simulation_mean_median/` | Per-simulation node-property mean/median distributions compared across boundary angles for groups 0, 1, and rest. |
| `boundary_distributions/edge_simulation_mean_median/` | The corresponding particle-contact comparisons. |
| `boundary_distributions/*_force_split_histograms/` | Force-cluster versus non-force histograms under both requested boundary grouping schemes. |
| `boundary_distributions/*_force_split_simulation_mean_median/` | Force/non-force per-simulation mean/median distributions across boundary angles. |
| `nodes_with_boundary_distance.csv.gz` | Node features plus exact graph distance and grouped shell. |
| `edges_with_boundary_distance.csv.gz` | Edge features plus exact distance and grouped shell. |
| `*_property_summary_by_shell.csv` | Counts, mean, median, standard deviation, P5, and P95. |
| `*_boundary_contrast_tests.csv` | Pooled KS and Wasserstein results for the two requested contrasts within each geometry. |
| `*_simulation_mean_median_values.csv` | One mean and median per simulation/property/group: the replicate-level comparison data. |
| `*_simulation_angle_tests.csv` | Pairwise Mann-Whitney angle tests on simulation means/medians, with global Benjamini-Hochberg correction. |
| `*_force_split_*` CSV files | Force/non-force pooled summaries/tests and simulation-replicate values/tests. |
| `*_color_limits.csv` | Common P5/P95 color limits used by every geometry panel for each property. |
| `boundary_assignment_audit.csv` | Per-simulation shell counts and unreachable-node check. |
| `analysis_metadata.json` | Inputs, conventions, properties, and representative simulations. |
{audit_inventory}

## Conventions

- Boundary node (distance 0): a particle incident to a contact where `is_wall_contact=True`.
- Node distance: shortest number of particle-particle contacts from a boundary particle.
- Edge distance: the smaller distance of its two particle endpoints. Thus a particle-particle contact incident to a boundary particle is edge distance 0.
- Wall placeholders and wall-contact edges are used only to identify boundary particles. They are excluded from all analyzed tables, plots, distributions, and tests.
- Displayed groups: 0 (boundary), 1, and rest (distance >=2 or unreachable).
- Contrast 1 is distance 0 versus every remaining entity (distance >=1 or unreachable).
- Comparison 2 shows three separate categories: distance 0, distance 1, and rest (distance >=2 or unreachable). Its CSV reports all three pairwise tests.
- The same P5/P95 color range is used for all geometry panels of a property. Values outside it are clipped only for color mapping, never in saved tables.
- Distribution x-limits use P1/P99 for readability; summary and tests use all finite values.
- Histogram contrasts are pooled descriptive views. Angle inference uses per-simulation means/medians so particles or contacts are not treated as independent replicates.
- 3-D property colors are restricted to the stored force labels (`is_force_chain_node` for nodes and `is_high_force` for contacts); non-force entities remain light gray.
- The same stored force labels define the force/non-force distribution split. No force labels or graph properties are recomputed.
- Representative simulations: `{json.dumps(representative, sort_keys=True)}`.

## Inventory

- Node properties ({len(node_props)}): {', '.join(f'`{x}`' for x in node_props)}
- Edge properties ({len(edge_props)}): {', '.join(f'`{x}`' for x in edge_props)}

## Reproduce

From the repository root:

```bash
python AnalysisScripts/PostAnalysis/analyze_boundary_layers.py
```

Use `--datasets FinalLoadState` (or the periodic dataset key) to run one target. Use `--lower-percentile 10 --upper-percentile 90` for P10/P90 color clipping.
"""
    (output / "README.md").write_text(text)


def analyze(name: str, source: Path, lower: float, upper: float, skip_3d: bool = False) -> None:
    output = source / "PropertyBoundaryAnalysis"
    output.mkdir(parents=True, exist_ok=True)
    for obsolete in ("node_geometry_tests_by_shell.csv", "edge_geometry_tests_by_shell.csv"):
        (output / obsolete).unlink(missing_ok=True)
    nodes = pd.read_csv(source / "node_features.csv")
    edges = pd.read_csv(source / "edge_features.csv")
    edges["node1"] = edges["node1"].map(normalize_endpoint)
    edges["node2"] = edges["node2"].map(normalize_endpoint)
    node_props = scalar_properties(nodes, NODE_EXCLUDE)
    edge_props = scalar_properties(edges, EDGE_EXCLUDE)
    nodes, edges, audit = assign_shells(nodes, edges)
    node_limits = color_limits(nodes, node_props, lower, upper)
    edge_limits = color_limits(edges, edge_props, lower, upper)
    representative = representative_simulations(nodes)
    box_lengths, periodic_axes = {}, []
    geometry_path = source.parent / "ReusablePipeline" / "job0_metadata" / "geometry_estimate.json"
    if geometry_path.exists():
        geometry = json.loads(geometry_path.read_text())
        box_lengths = {int(axis): float(length) for axis, length in geometry.get("box_lengths", {}).items()}
        periodic_axes = [int(axis) for axis in geometry.get("periodic_axes", [])]
    if not skip_3d:
        plot_3d_maps(nodes, edges, node_props, "node", node_limits, output / "node_properties_3d", representative,
                     box_lengths=box_lengths, periodic_axes=periodic_axes)
        plot_3d_maps(nodes, edges, edge_props, "edge", edge_limits, output / "edge_properties_3d", representative,
                     box_lengths=box_lengths, periodic_axes=periodic_axes)
    dist = output / "boundary_distributions"; dist.mkdir(exist_ok=True)
    ns, nt = distribution_outputs(nodes, node_props, "node", dist)
    es, et = distribution_outputs(edges, edge_props, "edge", dist)
    nsv, nst = simulation_summary_outputs(nodes, node_props, "node", dist)
    esv, est = simulation_summary_outputs(edges, edge_props, "edge", dist)
    nfhs, nfht = force_split_histogram_outputs(nodes, node_props, "node", dist)
    efhs, efht = force_split_histogram_outputs(edges, edge_props, "edge", dist)
    nfsv, nfst = force_split_simulation_outputs(nodes, node_props, "node", dist)
    efsv, efst = force_split_simulation_outputs(edges, edge_props, "edge", dist)
    nodes.to_csv(output / "nodes_with_boundary_distance.csv.gz", index=False)
    edges.to_csv(output / "edges_with_boundary_distance.csv.gz", index=False)
    pd.DataFrame(audit).to_csv(output / "boundary_assignment_audit.csv", index=False)
    node_limits.to_csv(output / "node_property_color_limits.csv", index=False)
    edge_limits.to_csv(output / "edge_property_color_limits.csv", index=False)
    ns.to_csv(output / "node_property_summary_by_shell.csv", index=False)
    es.to_csv(output / "edge_property_summary_by_shell.csv", index=False)
    nt.to_csv(output / "node_boundary_contrast_tests.csv", index=False)
    et.to_csv(output / "edge_boundary_contrast_tests.csv", index=False)
    nsv.to_csv(output / "node_simulation_mean_median_values.csv", index=False)
    esv.to_csv(output / "edge_simulation_mean_median_values.csv", index=False)
    nst.to_csv(output / "node_simulation_angle_tests.csv", index=False)
    est.to_csv(output / "edge_simulation_angle_tests.csv", index=False)
    nfhs.to_csv(output / "node_force_split_pooled_summary.csv", index=False)
    efhs.to_csv(output / "edge_force_split_pooled_summary.csv", index=False)
    nfht.to_csv(output / "node_force_split_pooled_tests.csv", index=False)
    efht.to_csv(output / "edge_force_split_pooled_tests.csv", index=False)
    nfsv.to_csv(output / "node_force_split_simulation_values.csv", index=False)
    efsv.to_csv(output / "edge_force_split_simulation_values.csv", index=False)
    nfst.to_csv(output / "node_force_split_simulation_tests.csv", index=False)
    efst.to_csv(output / "edge_force_split_simulation_tests.csv", index=False)
    metadata = {"dataset": name, "source": str(source), "boundary_definition": "particle incident to is_wall_contact edge",
                "node_distance": "shortest particle-contact path from boundary", "edge_distance": "minimum endpoint node distance for particle-particle contacts",
                "analysis_view": "particle nodes and particle-particle contacts only; wall records used only to seed boundary",
                "force_visualization": "property color only on stored force nodes/edges; opposite force entity light red; background light gray",
                "visualization_periodic_axes": periodic_axes, "visualization_box_lengths": box_lengths,
                "periodic_edge_rendering": "minimum-image displacement split at periodic faces" if periodic_axes else "direct particle-center segment",
                "shells": SHELL_ORDER, "contrasts": [x[0] for x in CONTRASTS],
                "views": [x[0] for x in VIEWS], "color_percentiles": [lower, upper], "representative_simulations": representative,
                "node_properties": node_props, "edge_properties": edge_props}
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_readme(output, name, source, node_props, edge_props, representative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--lower-percentile", type=float, default=5)
    parser.add_argument("--upper-percentile", type=float, default=95)
    parser.add_argument("--skip-3d", action="store_true", help="Regenerate tables/distributions without rerendering existing 3-D figures")
    args = parser.parse_args()
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("percentiles must satisfy 0 <= lower < upper <= 100")
    for name in args.datasets:
        analyze(name, DATASETS[name], args.lower_percentile, args.upper_percentile, skip_3d=args.skip_3d)


if __name__ == "__main__":
    main()
