#!/usr/bin/env python3
"""Where betweenness and force clusters sit in the specimen, and a height-controlled test.

1) r-z maps per angle, binned exactly as the manuscript betweenness maps
   (ManuscriptVisualization/plot_final_load_betweenness_rz_bins.py: pooled systems,
   13 x 10 cells, cells inclined along the boundary for non-zero angles):
   mean betweenness, high-force share, and mean number of high-force
   particle-particle contacts per particle (high_force_degree).
2) Height layers: each system is cut into equal-count layers of the
   boundary-following height z_eff (the rows of the maps; plain z at 0deg).
   Betweenness is ranked within each layer, and the high-force share among the
   layer's top k% is compared with that layer's own high-force share. Pooling the
   layers gives observed / expected high-force particles in the top k%, with the
   expectation taken layer by layer (height-controlled) or from the whole system
   (global, as in step 02). If the angle trend survives height control, it is not
   explained by force clusters and high betweenness occupying different heights.
Systems are the replicates: mean and 95% t interval over the systems of an angle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_only_common import (OUTPUT_ROOT, PARTICLES_CSV, angle_colors, angle_key, load_particles,
                              mean_ci, sorted_angles, summarize)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ManuscriptVisualization"))
from plot_final_load_betweenness_rz_bins import SPLIT_R, bin_rz, z_eff_from_rz  # noqa: E402

OUT = OUTPUT_ROOT / "04_spatial_comparison"
M_TO_MM = 1000.0
MAP_QUANTITIES = {
    "betweenness": ("Mean betweenness", (0.005, 0.045)),
    "is_high_force": ("High-force share", None),
    "high_force_degree": ("High-force contacts per particle", None),
}


def rz_maps(particles, angles):
    maps, rows = {}, []
    for angle in angles:
        sub = particles[particles.angle == angle]
        positions = sub[["x", "y", "z"]].to_numpy(float)
        for quantity in MAP_QUANTITIES:
            binned = bin_rz(positions, sub[quantity].to_numpy(float), angle_key(angle), SPLIT_R)
            maps[angle, quantity] = binned
            for iz, ir in np.ndindex(binned["mean"].shape):
                rows.append({"angle": angle, "quantity": quantity, "z_bin": iz, "r_bin": ir,
                             "particles": int(binned["counts"][iz, ir]), "mean": binned["mean"][iz, ir]})
    return maps, pd.DataFrame(rows)


def map_correlations(table, angles):
    """Spearman rho across occupied cells between mean betweenness and each force map."""
    rows = []
    wide = table.pivot_table(index=["angle", "z_bin", "r_bin"], columns="quantity", values="mean")
    for angle in angles:
        cells = wide.loc[angle].dropna()
        for quantity in ("is_high_force", "high_force_degree"):
            rows.append({"angle": angle, "force_map": quantity, "cells": len(cells),
                         "spearman_rho_with_betweenness": cells.betweenness.rank().corr(cells[quantity].rank())})
    return pd.DataFrame(rows)


def plot_maps(maps, angles, path):
    r_max = max(m["r_mesh"].max() for m in maps.values()) * M_TO_MM
    z_max = max(m["z_mesh"].max() for m in maps.values()) * M_TO_MM
    fig, axes = plt.subplots(len(MAP_QUANTITIES), len(angles), figsize=(3.3 * len(angles) + 1.2,
                             3.6 * len(MAP_QUANTITIES)), squeeze=False)
    for row, (quantity, (label, limits)) in enumerate(MAP_QUANTITIES.items()):
        if limits is None:
            values = np.concatenate([maps[a, quantity]["mean"].ravel() for a in angles])
            limits = tuple(np.nanpercentile(values, [2, 98]))
        for col, angle in enumerate(angles):
            ax = axes[row, col]
            m = maps[angle, quantity]
            mesh = ax.pcolormesh(m["r_mesh"] * M_TO_MM, m["z_mesh"] * M_TO_MM, m["mean"], shading="flat",
                                 cmap="coolwarm", vmin=limits[0], vmax=limits[1])
            ax.set_xlim(0, r_max)
            ax.set_ylim(z_max, 0)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(angle)
            if row == len(MAP_QUANTITIES) - 1:
                ax.set_xlabel("r (mm)")
            if col == 0:
                ax.set_ylabel("z (mm)")
        fig.colorbar(mesh, ax=list(axes[row]), label=label, extend="both", fraction=0.02, pad=0.02)
    fig.suptitle("Spatial maps: betweenness vs force clusters (pooled systems; core only)")
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def add_layers(particles, n_layers):
    framed = []
    for (angle, _), sub in particles.groupby(["angle", "sim_idx"]):
        sub = sub.copy()
        centered = sub[["x", "y", "z"]].to_numpy(float) - sub[["x", "y", "z"]].to_numpy(float).mean(axis=0)
        r = np.hypot(centered[:, 0], centered[:, 1])
        z = centered[:, 2] - centered[:, 2].min()
        sub["z_eff"] = z_eff_from_rz(r, z, math.tan(math.radians(angle_key(angle))), SPLIT_R)
        order = sub.z_eff.rank(method="first").to_numpy() - 1
        sub["layer"] = (order * n_layers // len(sub)).astype(int) + 1
        framed.append(sub)
    return pd.concat(framed, ignore_index=True)


def layer_tables(layered, top_percent):
    layer_rows, system_rows = [], []
    for (angle, sim), system in layered.groupby(["angle", "sim_idx"]):
        system_share = system.is_high_force.mean()
        system_mean_b = system.betweenness.mean()
        observed = expected_layer = n_top_total = 0.0
        for layer, sub in system.groupby("layer"):
            top_fraction = (sub.betweenness.rank(ascending=False, method="average") - 0.5) / len(sub)
            top = sub[top_fraction < top_percent / 100]
            layer_share = sub.is_high_force.mean()
            observed += top.is_high_force.sum()
            expected_layer += len(top) * layer_share
            n_top_total += len(top)
            layer_rows.append({
                "angle": angle, "sim_idx": sim, "layer": layer, "particles": len(sub),
                "high_force_share_rel_system": layer_share / system_share,
                "mean_betweenness_rel_system": sub.betweenness.mean() / system_mean_b,
                "top_particles": len(top),
                "within_layer_top_enrichment": top.is_high_force.mean() / layer_share if layer_share else np.nan,
            })
        system_rows.append({
            "angle": angle, "sim_idx": sim, "top_particles": int(n_top_total),
            "observed_high_force_in_top": int(observed),
            "obs_over_exp_height_controlled": observed / expected_layer,
            "obs_over_exp_global": observed / (n_top_total * system_share),
        })
    return pd.DataFrame(layer_rows), pd.DataFrame(system_rows)


def plot_layers(layers, angles, n_layers, top_percent, path):
    colors = angle_colors(angles)
    panels = (("high_force_share_rel_system", "High-force share / system share"),
              ("mean_betweenness_rel_system", "Mean betweenness / system mean"),
              ("within_layer_top_enrichment", f"High-force enrichment in layer's top {top_percent:g}%"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.3), squeeze=False)
    for ax, (column, title) in zip(axes[0], panels):
        for angle in angles:
            s = summarize(layers[layers.angle == angle], ["layer"], [column])
            ax.plot(s.layer, s[f"{column}_mean"], marker="o", color=colors[angle], label=angle)
            ax.fill_between(s.layer, s[f"{column}_ci_low"], s[f"{column}_ci_high"],
                            color=colors[angle], alpha=0.18)
        ax.axhline(1, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, n_layers + 1))
        ax.set_xlabel("Height layer (1 = lowest z_eff; top row of the r-z maps)")
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(title="Angle", fontsize=8)
    fig.suptitle(f"Height layers ({n_layers} equal-count layers per system; band = 95% CI over systems)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_height_control(systems, angles, top_percent, path):
    styles = {"obs_over_exp_global": ("Global (whole-system expectation)", "#7F7F7F", -0.12),
              "obs_over_exp_height_controlled": ("Height-controlled (layer expectation)", "#D95F02", 0.12)}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for column, (label, color, offset) in styles.items():
        stats = []
        for i, angle in enumerate(angles):
            values = systems[systems.angle == angle][column]
            ax.scatter(i + offset + rng.uniform(-0.03, 0.03, len(values)), values, s=10, color=color,
                       alpha=0.35, linewidths=0)
            stats.append(mean_ci(values))
        stats = np.array(stats)
        ax.errorbar(np.arange(len(angles)) + offset, stats[:, 0],
                    yerr=[stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]], color=color, marker="o",
                    capsize=3, linewidth=1.5, label=label)
    ax.axhline(1, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels(angles)
    ax.set_ylabel(f"High-force particles in top {top_percent:g}%: observed / expected")
    ax.legend(fontsize=8)
    ax.set_title("Does the angle trend survive controlling for height?", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=5, help="Equal-count height layers per system.")
    parser.add_argument("--top-percent", type=float, default=20.0,
                        help="Top share of each layer's betweenness ranking.")
    args = parser.parse_args()

    particles = load_particles()
    if not {"x", "y", "z"} <= set(particles.columns):
        raise KeyError("particles.csv lacks positions; rerun 00_export_workbook_labels.py")
    particles["is_high_force"] = (particles.group == "high_force").astype(float)
    angles = sorted_angles(particles.angle)

    tables, figures = OUT / "tables", OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    maps, map_table = rz_maps(particles, angles)
    map_table.to_csv(tables / "rz_bins.csv", index=False)
    correlations = map_correlations(map_table, angles)
    correlations.to_csv(tables / "rz_map_correlations.csv", index=False)
    plot_maps(maps, angles, figures / "rz_maps_betweenness_vs_force.png")

    layered = add_layers(particles, args.layers)
    layers, systems = layer_tables(layered, args.top_percent)
    layers.to_csv(tables / "layers_per_system.csv", index=False)
    systems.to_csv(tables / "height_controlled_enrichment_per_system.csv", index=False)
    summary = summarize(systems, ["angle"], ["obs_over_exp_global", "obs_over_exp_height_controlled"])
    summary.to_csv(tables / "height_controlled_enrichment_summary.csv", index=False)
    summarize(layers, ["angle", "layer"], ["high_force_share_rel_system", "mean_betweenness_rel_system",
                                           "within_layer_top_enrichment"]).to_csv(
        tables / "layers_summary.csv", index=False)
    plot_layers(layers, angles, args.layers, args.top_percent, figures / "height_layer_profiles.png")
    plot_height_control(systems, angles, args.top_percent, figures / "height_controlled_enrichment.png")

    (OUT / "metadata.json").write_text(json.dumps({
        "input": str(PARTICLES_CSV),
        "rz_binning": "imported from ManuscriptVisualization/plot_final_load_betweenness_rz_bins.py",
        "split_r_m": SPLIT_R, "layers": args.layers, "top_percent": args.top_percent,
        "z_eff": "boundary-following height (z minus k*r, dual slope); equals z at 0deg",
        "observed_over_expected": "sum over layers of high-force particles in each layer's top k%, "
                                  "divided by the count expected from layer shares (height-controlled) "
                                  "or from the system share (global)",
        "replicates": "systems; mean and 95% t interval over systems per angle",
    }, indent=2))
    print(correlations.round(3).to_string(index=False))
    print(summary[["angle", "obs_over_exp_global_mean", "obs_over_exp_global_ci_low",
                   "obs_over_exp_global_ci_high", "obs_over_exp_height_controlled_mean",
                   "obs_over_exp_height_controlled_ci_low",
                   "obs_over_exp_height_controlled_ci_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
