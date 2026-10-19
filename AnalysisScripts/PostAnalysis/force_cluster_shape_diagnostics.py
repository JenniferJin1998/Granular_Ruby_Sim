#!/usr/bin/env python3
"""Separate particle-center elongation and planarity in saved force clusters."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force_threshold_percolation import atomic_csv


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "AnalysisResults/periodic_boundaries/2026-08-03/2_force_cluster_comparison/force_split2/connected_force_clusters"
COLORS = {"0deg": "#0072B2", "30deg": "#D55E00"}


def shape_metrics(eigenvalues):
    values = np.asarray(eigenvalues, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values, axis=1) > 0):
        raise ValueError("Expected finite nonnegative descending eigenvalues")
    largest, intermediate, smallest = values.T
    tolerance = 1e-10 * largest
    valid = largest > 0
    middle_valid = valid & (intermediate > tolerance)
    small_valid = valid & (smallest > tolerance)
    elongation = np.sqrt(np.divide(largest, intermediate, out=np.full_like(largest, np.nan), where=middle_valid))
    flatness = np.sqrt(np.divide(intermediate, smallest, out=np.full_like(largest, np.nan), where=small_valid))
    linearity = np.divide(largest-intermediate, largest, out=np.full_like(largest, np.nan), where=valid)
    planarity = np.divide(intermediate-smallest, largest, out=np.full_like(largest, np.nan), where=valid)
    sphericity = np.divide(smallest, largest, out=np.full_like(largest, np.nan), where=valid)
    return pd.DataFrame(dict(elongation_ratio=elongation, flatness_ratio=flatness,
                             linearity=linearity, planarity=planarity, sphericity=sphericity,
                             spatial_rank=np.sum(values > tolerance[:, None], axis=1)))


def save(figure, path):
    for extension in ("png", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def run():
    output = ROOT / "shape_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    clusters = pd.read_csv(ROOT / "tables/force_clusters.csv")
    metrics = shape_metrics(clusters[[f"gyration_eigenvalue_{index}" for index in (1, 2, 3)]].to_numpy())
    frame = pd.concat([clusters, metrics], axis=1)
    if not np.allclose(frame.linearity + frame.planarity + frame.sphericity, 1):
        raise ValueError("Shape decomposition does not sum to one")
    records = []
    selections = {"all_clusters": frame, "at_least_4_particles": frame[frame.node_count >= 4]}
    for selection, selected in selections.items():
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for axis, metric, title in zip(axes, ["elongation_ratio", "flatness_ratio"],
                                        [r"Elongation $E=\sqrt{\lambda_1/\lambda_2}$", r"Flatness $F=\sqrt{\lambda_2/\lambda_3}$"]):
            finite = selected[metric].dropna()
            edges = np.geomspace(1, max(float(finite.max()), 1.01), 40)
            for angle, color in COLORS.items():
                group = selected[selected.angle == angle]
                values = group[metric].dropna()
                axis.hist(values, bins=edges, weights=np.ones(len(values))/len(values), histtype="step", linewidth=1.8,
                          label=f"{angle}: {len(values)}/{len(group)} finite", color=color)
                records.append(dict(selection=selection, angle=angle, metric=metric, total=len(group),
                                    finite=len(values), undefined=int(group[metric].isna().sum()),
                                    median=values.median(), p95=values.quantile(.95), maximum=values.max()))
            axis.set(xscale="log", xlabel=title, ylabel="Fraction of finite clusters per logarithmic bin")
            axis.legend(fontsize=9)
        figure.suptitle(selection.replace("_", " ").capitalize())
        save(figure, output / f"elongation_and_flatness_{selection}")
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True, sharey=True)
        for axis, metric in zip(axes, ["linearity", "planarity", "sphericity"]):
            for angle, color in COLORS.items():
                group = selected[selected.angle == angle]
                values = group[metric].dropna()
                axis.hist(values, bins=np.linspace(0, 1, 31), weights=np.ones(len(values))/len(values),
                          histtype="step", linewidth=1.8, color=color, label=angle)
                records.append(dict(selection=selection, angle=angle, metric=metric, total=len(group),
                                    finite=len(values), undefined=int(group[metric].isna().sum()),
                                    median=values.median(), p95=values.quantile(.95), maximum=values.max()))
            axis.set(xlabel=metric.capitalize(), xlim=(0, 1))
            axis.legend()
        axes[0].set_ylabel("Fraction of clusters per bin")
        figure.suptitle(selection.replace("_", " ").capitalize())
        save(figure, output / f"bounded_shape_measures_{selection}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True)
    for axis, angle in zip(axes, COLORS):
        subset = frame[(frame.angle == angle) & (frame.node_count >= 4)]
        artist = axis.scatter(subset.linearity, subset.planarity, c=np.log10(subset.node_count),
                              s=14, alpha=.6, cmap="viridis", vmin=np.log10(4), vmax=np.log10(frame.node_count.max()))
        axis.plot([0, 1], [1, 0], "--", color="gray", linewidth=.8)
        axis.set(xlabel="Linearity (chain-like → 1)", ylabel="Planarity (sheet-like → 1)", title=f"{angle}, at least 4 particles", xlim=(0, 1), ylim=(0, 1))
    figure.colorbar(artist, ax=axes, label="log10(particle count)")
    save(figure, output / "linearity_vs_planarity")
    atomic_csv(output / "cluster_shape_metrics.csv", frame)
    atomic_csv(output / "distribution_summary.csv", pd.DataFrame(records))
    atomic_csv(output / "simulation_mean_shape_metrics.csv", frame.groupby(["angle", "sim_idx"])[list(metrics.columns)].mean().reset_index())
    (output / "README.md").write_text(Path(__file__).with_suffix(".md").read_text())
    print(f"Saved shape diagnostics for {len(frame)} clusters to {output}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"], "font.size": 12, "svg.fonttype": "none"})
    run()
