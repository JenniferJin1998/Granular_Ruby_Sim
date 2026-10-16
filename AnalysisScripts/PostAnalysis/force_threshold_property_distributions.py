#!/usr/bin/env python3
"""Sweep a pooled 0-degree reference force and summarize particle-only populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yfjin/threshold-property-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


PROJECT = Path(__file__).resolve().parents[2]
CASE = PROJECT / "AnalysisResults/final_load"
INPUT = CASE / "0_graph_and_basic_stats/force_splits/force_split2"
OUTPUT = CASE / "2_force_cluster_comparison/force_threshold_property_distributions"
GEOMETRIES = ("0deg", "15deg", "30deg", "45deg")
PROPERTIES = {"degree": "Degree", "closeness": "Closeness",
              "betweenness": "Betweenness", "angle_with_zz": "Contact angle to loading z (°)"}
NODE_PROPERTIES = tuple(name for name in PROPERTIES if name != "angle_with_zz")
GROUPS = ("high_force", "non_high_force")
COLORS = {"high_force": "#d95f02", "non_high_force": "#2474b5"}
LABELS = {"high_force": "High force", "non_high_force": "Non-high force"}


def threshold_grid(start, stop, step):
    if not np.isfinite([start, stop, step]).all() or start < 0 or stop <= start or step <= 0:
        raise ValueError("Require finite 0 <= start < stop and step > 0")
    intervals = (stop - start) / step
    if not np.isclose(intervals, round(intervals), rtol=0, atol=1e-9):
        raise ValueError("Threshold interval must be divisible by the step")
    return np.round(np.linspace(start, stop, int(round(intervals)) + 1), 12)


def particle_inputs(input_dir):
    nodes = pd.read_csv(input_dir / "node_features.csv")
    edges = pd.read_csv(input_dir / "edge_features.csv")
    for frame, wall_flag in ((nodes, "is_wall"), (edges, "is_wall_contact")):
        if frame[wall_flag].isna().any() or not frame[wall_flag].isin([True, False]).all():
            raise ValueError(f"Invalid {wall_flag} values")
    nodes = nodes.loc[~nodes.is_wall, ["geometry", "sim_idx", "node_id", *NODE_PROPERTIES]].copy()
    edges = edges.loc[~edges.is_wall_contact,
                      ["geometry", "sim_idx", "node1", "node2", "is_core_edge", "normal_force", "angle_with_zz"]].copy()
    if not edges.is_core_edge.all():
        raise ValueError("Particle-only input contains non-core edges")
    edges = edges.drop(columns="is_core_edge").rename(columns={"node1": "edge_u", "node2": "edge_v"})
    keys = ["geometry", "sim_idx", "node_id"]
    for column in ("edge_u", "edge_v"):
        edges[column] = pd.to_numeric(edges[column], errors="raise").astype(int)
    nodes = nodes.sort_values(keys, ignore_index=True)
    edges = edges.sort_values(["geometry", "sim_idx", "edge_u", "edge_v"], ignore_index=True)
    expected = {(geometry, sim_idx) for geometry in GEOMETRIES for sim_idx in range(20)}
    for frame in (nodes, edges):
        if set(zip(frame.geometry, frame.sim_idx)) != expected:
            raise ValueError("Expected four geometries with simulations 0 through 19")
    if nodes.duplicated(keys).any():
        raise ValueError("Duplicate particle identifiers")
    if edges.duplicated(["geometry", "sim_idx", "edge_u", "edge_v"]).any():
        raise ValueError("Duplicate contact identifiers")
    if not np.isfinite(nodes[list(NODE_PROPERTIES)].to_numpy(float)).all():
        raise ValueError("Nonfinite particle properties")
    if not np.isfinite(edges[["normal_force", "angle_with_zz"]].to_numpy(float)).all():
        raise ValueError("Nonfinite contact properties")
    if (edges.normal_force < 0).any() or not edges.angle_with_zz.between(0, 90).all():
        raise ValueError("Invalid normal forces or acute contact angles")
    reference_mean = float(edges.loc[edges.geometry.eq("0deg"), "normal_force"].mean())
    if reference_mean <= 0:
        raise ValueError("Reference mean must be positive")
    node_index = pd.MultiIndex.from_frame(nodes[keys])
    endpoints = []
    for column in ("edge_u", "edge_v"):
        identifiers = edges[["geometry", "sim_idx", column]].rename(columns={column: "node_id"})
        indices = node_index.get_indexer(pd.MultiIndex.from_frame(identifiers))
        if (indices < 0).any():
            raise ValueError("Missing particle endpoint")
        endpoints.append(indices)
    incidence = np.bincount(np.concatenate(endpoints), minlength=len(nodes))
    if not np.array_equal(incidence, nodes.degree.to_numpy()):
        raise ValueError("Stored degree disagrees with particle-only graph topology")
    max_incident_force = np.full(len(nodes), -np.inf)
    for indices in endpoints:
        np.maximum.at(max_incident_force, indices, edges.normal_force.to_numpy())
    nodes["max_incident_particle_contact_force"] = max_incident_force
    return nodes, edges, reference_mean


def population_summaries(nodes, edges, reference_mean, multipliers, per_simulation):
    rows = []
    grouping = ["geometry", "sim_idx"] if per_simulation else ["geometry"]
    for scope, frame, properties, force_column in (
        ("node", nodes, NODE_PROPERTIES, "max_incident_particle_contact_force"),
        ("edge", edges, ("angle_with_zz",), "normal_force"),
    ):
        for key, population in frame.groupby(grouping, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            identifiers = dict(zip(grouping, key))
            forces = population[force_column].to_numpy(float)
            values = {prop: population[prop].to_numpy(float) for prop in properties}
            for multiplier in multipliers:
                cutoff = float(multiplier * reference_mean)
                high_mask = forces >= cutoff
                for group, mask in (("high_force", high_mask), ("non_high_force", ~high_mask)):
                    count = int(mask.sum())
                    for prop, property_values in values.items():
                        selected = property_values[mask]
                        rows.append({**identifiers, "n": float(multiplier), "force_threshold": cutoff,
                                     "scope": scope, "property": prop, "group": group,
                                     "count": count, "population_count": len(population),
                                     "fraction": count / len(population),
                                     "mean": float(selected.mean()) if count else np.nan,
                                     "median": float(np.median(selected)) if count else np.nan})
    return pd.DataFrame(rows)


def validate_summaries(pooled, simulations):
    keys = ["geometry", "n", "scope", "property"]
    for frame, extra in ((pooled, []), (simulations, ["sim_idx"])):
        grouped = frame.groupby(keys + extra, sort=False)
        totals = grouped["count"].sum()
        expected = grouped["population_count"].first()
        if not totals.equals(expected):
            raise ValueError("High/non-high counts do not partition their population")
        empty = frame["count"].eq(0)
        if not frame.loc[empty, ["mean", "median"]].isna().all().all():
            raise ValueError("Empty groups must have undefined summaries")
        if not np.isfinite(frame.loc[~empty, ["mean", "median"]].to_numpy()).all():
            raise ValueError("Nonempty groups must have finite summaries")
    by_group = keys + ["group"]
    expected_counts = simulations.groupby(by_group)["count"].sum().sort_index()
    actual_counts = pooled.set_index(by_group)["count"].sort_index()
    if not np.array_equal(expected_counts.to_numpy(), actual_counts.to_numpy()):
        raise ValueError("Pooled and per-simulation population counts disagree")
    weighted = simulations.assign(weighted_sum=simulations["mean"].fillna(0) * simulations["count"])
    sums = weighted.groupby(by_group).agg(weighted_sum=("weighted_sum", "sum"), count=("count", "sum"))
    expected_means = (sums.weighted_sum / sums["count"].replace(0, np.nan)).sort_index()
    actual_means = pooled.set_index(by_group)["mean"].sort_index()
    if not np.allclose(expected_means, actual_means, rtol=1e-12, atol=1e-14, equal_nan=True):
        raise ValueError("Pooled means disagree with count-weighted simulation means")
    for _, frame in pooled.groupby(["geometry", "scope", "property", "group"]):
        changes = np.diff(frame.sort_values("n")["count"])
        if ((changes > 0).any() if frame.group.iloc[0] == "high_force" else (changes < 0).any()):
            raise ValueError("Population counts are not monotone with threshold")


def save_figure(fig, destination, formats, dpi):
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = destination.with_suffix("." + extension)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def property_axes(axes, statistic):
    for axis, (prop, label) in zip(axes.flat, PROPERTIES.items()):
        scope = "particles" if prop in NODE_PROPERTIES else "contacts"
        axis.set_title(f"{label} · {scope}")
        axis.set_ylabel(f"Pooled {statistic}")
        axis.grid(alpha=0.2)


def make_plots(summary, output, multipliers, formats, dpi):
    paths = []
    geometry_x = np.arange(len(GEOMETRIES))
    geometry_labels = [geometry.replace("deg", "°") for geometry in GEOMETRIES]
    normalizer = Normalize(float(multipliers.min()), float(multipliers.max()))
    palette = matplotlib.colormaps["viridis"]
    for statistic in ("mean", "median"):
        for geometry in GEOMETRIES:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
            property_axes(axes, statistic)
            for axis, prop in zip(axes.flat, PROPERTIES):
                frame = summary.loc[summary.geometry.eq(geometry) & summary.property.eq(prop)]
                for group in GROUPS:
                    curve = frame.loc[frame.group.eq(group)].sort_values("n")
                    axis.plot(curve.n, curve[statistic], color=COLORS[group], label=LABELS[group], lw=1.6)
                axis.set_xlabel("Threshold multiplier n (fixed 0° reference mean)")
                axis.set_xlim(multipliers.min(), multipliers.max())
                axis.axvline(2, color="#777777", ls=":", lw=0.8)
                axis.legend(frameon=False)
            fig.suptitle(f"Final load · {geometry.replace('deg', '°')} · pooled {statistic}")
            paths.extend(save_figure(fig, output / "threshold_curves" / statistic / geometry, formats, dpi))
        for group in GROUPS:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
            property_axes(axes, statistic)
            for axis, prop in zip(axes.flat, PROPERTIES):
                frame = summary.loc[summary.group.eq(group) & summary.property.eq(prop)]
                grid = frame.pivot(index="n", columns="geometry", values=statistic).reindex(index=multipliers, columns=GEOMETRIES)
                for multiplier, row in grid.iterrows():
                    emphasized = np.isclose(multiplier, round(multiplier), rtol=0, atol=1e-10)
                    axis.plot(geometry_x, row.to_numpy(), color=palette(normalizer(multiplier)),
                              lw=1.6 if emphasized else 0.65, alpha=1 if emphasized else 0.3,
                              marker="o" if emphasized else None, markersize=3)
                axis.set_xticks(geometry_x, geometry_labels)
                axis.set_xlabel("Geometry")
            fig.colorbar(ScalarMappable(norm=normalizer, cmap=palette), ax=list(axes.flat),
                         label="Threshold multiplier n", shrink=0.9)
            fig.suptitle(f"{LABELS[group]} · pooled {statistic} · all {len(multipliers)} thresholds")
            paths.extend(save_figure(fig, output / "geometry_lines" / statistic / group, formats, dpi))
        fig = plt.figure(figsize=(13, 10))
        fig.subplots_adjust(left=0.02, right=0.94, bottom=0.10, top=0.93,
                            wspace=0.15, hspace=0.15)
        for index, (prop, label) in enumerate(PROPERTIES.items(), start=1):
            axis = fig.add_subplot(2, 2, index, projection="3d")
            geometry_grid, threshold_mesh = np.meshgrid(geometry_x, multipliers)
            for group in GROUPS:
                frame = summary.loc[summary.group.eq(group) & summary.property.eq(prop)]
                values = frame.pivot(index="n", columns="geometry", values=statistic).reindex(index=multipliers, columns=GEOMETRIES).to_numpy()
                axis.plot_surface(geometry_grid, threshold_mesh, values, color=COLORS[group],
                                  alpha=0.5, linewidth=0.1, edgecolor=COLORS[group],
                                  rcount=len(multipliers), ccount=len(GEOMETRIES), shade=False)
            axis.set_xticks(geometry_x, geometry_labels)
            axis.set_xlabel("Geometry", labelpad=8)
            axis.set_ylabel("Threshold n", labelpad=8)
            axis.set_zlabel(f"Pooled {statistic}", labelpad=9)
            axis.set_title(label)
            axis.view_init(elev=26, azim=-55)
        fig.legend(handles=[Line2D([], [], color=COLORS[group], lw=5, label=LABELS[group]) for group in GROUPS],
                   loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
        fig.suptitle(f"Final load · geometry × threshold · pooled {statistic}")
        paths.extend(save_figure(fig, output / "geometry_threshold_surfaces" / statistic, formats, dpi))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--stop", type=float, default=4)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    multipliers = threshold_grid(args.start, args.stop, args.step)
    if args.dpi < 1:
        parser.error("dpi must be positive")
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
                                "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    nodes, edges, reference_mean = particle_inputs(args.input_dir)
    print(f"Particle-only inputs: {len(nodes):,} nodes, {len(edges):,} contacts; reference mean {reference_mean:.16g}", flush=True)
    pooled = population_summaries(nodes, edges, reference_mean, multipliers, False)
    simulations = population_summaries(nodes, edges, reference_mean, multipliers, True)
    validate_summaries(pooled, simulations)
    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(tables / "pooled_geometry_group_summaries.csv", index=False)
    simulations.to_csv(tables / "per_simulation_group_summaries.csv", index=False)
    balanced = simulations.groupby(["geometry", "n", "force_threshold", "scope", "property", "group"], as_index=False).agg(
        simulations_with_group=("mean", "count"), mean_of_simulation_means=("mean", "mean"),
        mean_of_simulation_medians=("median", "mean"))
    balanced.to_csv(tables / "simulation_balanced_summaries.csv", index=False)
    input_nodes = nodes.copy()
    input_nodes["max_incident_particle_contact_force"] = input_nodes.max_incident_particle_contact_force.replace(-np.inf, np.nan)
    input_nodes.to_csv(tables / "particle_input_properties.csv", index=False)
    edges.to_csv(tables / "particle_contact_input_properties.csv", index=False)
    pd.DataFrame({"n": multipliers, "force_threshold": multipliers * reference_mean}).to_csv(tables / "thresholds.csv", index=False)
    print(f"Validated and saved {len(pooled):,} pooled and {len(simulations):,} per-simulation rows", flush=True)
    figures = [] if args.tables_only else make_plots(pooled, args.output_root / "figures", multipliers, args.formats, args.dpi)
    sources = {name: {"path": str((args.input_dir / name).resolve()),
                      "sha256": hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()}
               for name in ("node_features.csv", "edge_features.csv")}
    manifest = {"dataset": "final_load", "geometries": GEOMETRIES, "simulations_per_geometry": 20,
                "reference_geometry": "0deg", "reference_mean_normal_force": reference_mean,
                "reference_contact_count": int(edges.geometry.eq("0deg").sum()),
                "threshold_definition": "normal_force >= n * pooled 0deg particle-particle mean normal force; fixed across simulations and geometries",
                "multipliers": multipliers.tolist(), "node_rule": "High iff incident to at least one retained high-force particle-particle edge; all other particles are non-high",
                "walls": "Excluded from reference mean, recorded populations, node labels, and graph properties",
                "properties": "Stored complete particle-only graph degree/closeness/betweenness; stored acute contact-normal angle to z in degrees; properties held fixed during sweep",
                "plot_aggregation": "Particle/contact pooled mean or median within each geometry and threshold group; not mean of simulation summaries",
                "empty_groups": "count=0, mean/median=NaN; gaps in curves and surfaces",
                "surface_interpolation": "Surfaces connect sampled geometry/threshold summaries for visualization; geometry axis uses equally spaced categorical orientations",
                "limits": "Means/medians describe distribution location, not its complete shape; descriptive analysis without significance claims",
                "validation": "Particle degree checked against contact incidence; exhaustive disjoint partitions; monotone counts; pooled/per-simulation counts and weighted means agree",
                "input_counts": {"particles": len(nodes), "particle_particle_contacts": len(edges)},
                "sources": sources, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figure_files": figures}
    (args.output_root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_root / "README.md").write_text(
        "# Particle-only force-threshold property distributions\n\n"
        f"Reference mean: {reference_mean:.16g}; {len(multipliers)} multipliers from {multipliers[0]:g} to {multipliers[-1]:g}.\n\n"
        "See analysis_manifest.json for definitions, input hashes, and validation.\n"
        "The main plots use pooled means/medians. Per-simulation and simulation-balanced summaries are separate tables.\n"
        "Degree, closeness, and betweenness are fixed complete-core-graph properties; angle_with_zz is an edge property in degrees.\n"
        "Only group membership changes. Walls are excluded throughout. Empty groups remain undefined.\n"
        "Threshold curves: one four-property figure per geometry and statistic. Geometry lines: one figure per group and statistic; all thresholds are drawn, with integer n emphasized.\n"
        "3D surfaces: both groups together, connecting the sampled values. These plots describe location shifts, not full distribution shapes.\n"
    )
    print(f"Saved {len(figures)} figure files to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
