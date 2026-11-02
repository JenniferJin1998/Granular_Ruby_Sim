#!/usr/bin/env python3
"""Hydrostatic stress versus betweenness for high-force and non-high-force particles.

binned:  2 rows (high-force, non-high-force) x one column per angle. Each panel is
         normalized to the fraction of its particles; all panels share bin edges,
         axes, and a logarithmic color scale.
scatter: one panel per angle with both groups overlaid, drawn in random order so
         neither group systematically hides the other.
All 20 systems of an angle are pooled for display.

Both plots are also drawn with betweenness normalized within each system,
which removes system-to-system differences in overall betweenness level:
  system_mean_normalized: betweenness / that system's mean betweenness (1 = average);
  system_percentile:      rank within the system, (rank - 0.5) / n (1 = highest).
"""

from __future__ import annotations

import argparse
import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from core_only_common import (GROUP_COLORS, GROUP_LABELS, GROUPS, OUTPUT_ROOT, PARTICLES_CSV,
                              load_particles, sorted_angles)

OUT = OUTPUT_ROOT / "01_betweenness_vs_hydrostatic_stress"
Y_SCALE = 1e5
X_LABEL = "Betweenness centrality"
Y_LABEL = r"Hydrostatic stress ($\times 10^{-5}$)"
TITLE_SUFFIX = "(final load, force_split2, core only)"
# x column, axis label, and title tag for each version.
X_VERSIONS = {
    "": ("betweenness", X_LABEL, ""),
    "__system_mean_normalized": ("betweenness_rel_system_mean",
                                 "Betweenness / system mean betweenness", "normalized by system mean"),
    "__system_percentile": ("betweenness_system_percentile",
                            "Betweenness percentile within system", "percentile within system"),
}


def add_system_normalizations(particles):
    by_system = particles.groupby(["angle", "sim_idx"]).betweenness
    particles["betweenness_rel_system_mean"] = particles.betweenness / by_system.transform("mean")
    particles["betweenness_system_percentile"] = ((by_system.rank(method="average") - 0.5)
                                                  / by_system.transform("size"))
    return particles


def x_edges_for(values, column, bins):
    if column == "betweenness_system_percentile":
        return np.linspace(0.0, 1.0, bins + 1)
    return np.linspace(0.0, values.max(), bins + 1)


def plot_binned(particles, angles, x_edges, y_edges, x_column="betweenness", x_label=X_LABEL, tag=""):
    panels = {}
    for group in GROUPS:
        for angle in angles:
            sub = particles[(particles.angle == angle) & (particles.group == group)]
            counts = np.histogram2d(sub[x_column], sub.y_plot, bins=(x_edges, y_edges))[0]
            panels[group, angle] = (counts / counts.sum(), len(sub))
    positive = np.concatenate([p[p > 0] for p, _ in panels.values()])
    norm = LogNorm(positive.min(), positive.max())
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad("white")

    fig, axes = plt.subplots(len(GROUPS), len(angles), figsize=(4.0 * len(angles), 7.4),
                             sharex=True, sharey=True, squeeze=False)
    for row, group in enumerate(GROUPS):
        for col, angle in enumerate(angles):
            ax = axes[row, col]
            probability, n = panels[group, angle]
            image = ax.pcolormesh(x_edges, y_edges, np.ma.masked_less_equal(probability.T, 0),
                                  cmap=cmap, norm=norm)
            ax.set_title(f"{angle} | {GROUP_LABELS[group]} (n={n:,})", fontsize=10)
            if row == len(GROUPS) - 1:
                ax.set_xlabel(x_label)
            if col == 0:
                ax.set_ylabel(Y_LABEL)
    tag = f", {tag}" if tag else ""
    fig.suptitle(f"Hydrostatic stress versus betweenness{tag}: high-force vs non-high-force {TITLE_SUFFIX}")
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.08, top=0.9, wspace=0.08, hspace=0.18)
    cax = fig.add_axes((0.91, 0.15, 0.015, 0.68))
    fig.colorbar(image, cax=cax, label="Fraction of panel particles per bin (log color)")
    return fig, [float(positive.min()), float(positive.max())]


def plot_scatter(particles, angles, x_edges, y_edges, seed, x_column="betweenness", x_label=X_LABEL, tag=""):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(angles), figsize=(4.0 * len(angles), 4.2),
                             sharex=True, sharey=True, squeeze=False)
    for ax, angle in zip(axes[0], angles):
        sub = particles[particles.angle == angle]
        sub = sub.iloc[rng.permutation(len(sub))]
        ax.scatter(sub[x_column], sub.y_plot, c=sub.group.map(GROUP_COLORS), s=4, alpha=0.45,
                   linewidths=0, rasterized=True)
        counts = sub.group.value_counts()
        ax.set_title(f"{angle} (high n={counts.get('high_force', 0):,}, "
                     f"non-high n={counts.get('non_high_force', 0):,})", fontsize=10)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xlabel(x_label)
    axes[0, 0].set_ylabel(Y_LABEL)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=GROUP_COLORS[g],
                      label=GROUP_LABELS[g]) for g in GROUPS]
    axes[0, -1].legend(handles=handles, loc="lower right", frameon=True, fontsize=9)
    tag = f", {tag}" if tag else ""
    fig.suptitle(f"Hydrostatic stress versus betweenness{tag} {TITLE_SUFFIX}")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.84, wspace=0.08)
    return fig


def summary_table(particles, angles):
    rows = []
    for angle in angles:
        for group in GROUPS:
            sub = particles[(particles.angle == angle) & (particles.group == group)]
            rows.append({
                "angle": angle, "group": group, "particles": len(sub), "systems": sub.sim_idx.nunique(),
                "median_betweenness": sub.betweenness.median(),
                "median_stress_hydro": sub.stress_hydro.median(),
                "spearman_rho_pooled": sub.betweenness.rank().corr(sub.stress_hydro.rank()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261102, help="Scatter draw-order seed.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    particles = load_particles()
    particles["y_plot"] = particles.stress_hydro * Y_SCALE
    particles = add_system_normalizations(particles)
    angles = sorted_angles(particles.angle)
    x_edges = np.linspace(0.0, particles.betweenness.max(), args.bins + 1)
    y_edges = np.linspace(particles.y_plot.min(), max(particles.y_plot.max(), 0.0), args.bins + 1)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    figures, color_limits = {}, {}
    for suffix, (column, label, tag) in X_VERSIONS.items():
        edges = x_edges_for(particles[column], column, args.bins)
        figures[f"betweenness_vs_hydro__binned{suffix}"], color_limits[column] = plot_binned(
            particles, angles, edges, y_edges, column, label, tag)
        figures[f"betweenness_vs_hydro__scatter{suffix}"] = plot_scatter(
            particles, angles, edges, y_edges, args.seed, column, label, tag)
    for stem, figure in figures.items():
        for fmt in ("png", "pdf"):
            figure.savefig(OUT / "figures" / f"{stem}.{fmt}", dpi=args.dpi)
        plt.close(figure)

    import pandas as pd

    summary = pd.DataFrame(summary_table(particles, angles))
    summary.to_csv(OUT / "tables" / "group_summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "input": str(PARTICLES_CSV), "bins": args.bins,
        "x_range_betweenness": [x_edges[0], x_edges[-1]],
        "y_range_stress_hydro": [y_edges[0] / Y_SCALE, y_edges[-1] / Y_SCALE],
        "binned_normalization": "fraction of particles in each panel",
        "binned_log_color_limits": color_limits, "scatter_seed": args.seed,
        "system_normalizations": {
            "system_mean_normalized": "betweenness / mean betweenness of the particle's system",
            "system_percentile": "(average rank - 0.5) / n within the particle's system; 1 = highest"},
        "spearman_note": "pooled over systems; descriptive only (particles are not independent)",
    }, indent=2))
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
