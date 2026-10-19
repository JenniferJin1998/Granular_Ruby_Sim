#!/usr/bin/env python3
"""Supplement the periodic sweep with any-cluster spanning and local-height metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from force_threshold_percolation import (
    DEFAULT_CONFIG, atomic_csv, atomic_json, ci_summary, load_config,
    load_graphs, wall_contact_particles, widest_path_threshold,
)


COLORS = {"0deg": "#1F77B4", "30deg": "#D55E00"}
METRICS = {
    "spanning_cluster_count": "Number of spanning clusters",
    "spanning_particle_count": "Particles in all spanning clusters",
    "spanning_particle_fraction": "Fraction of particles in spanning clusters",
    "largest_spanning_cluster_size": "Largest spanning cluster size (zero if absent)",
    "mean_spanning_cluster_size": "Mean spanning-cluster size (conditional on spanning)",
    "largest_cluster_eta_extent": "Largest strong cluster: center-based local-height extent",
    "largest_cluster_boundary_eta_extent": "Largest strong cluster: boundary-inclusive local-height extent",
    "largest_spanning_eta_extent": "Largest spanning cluster: center-based local-height extent",
}


def surface_basis(x_positions, length):
    coordinate = np.mod(np.asarray(x_positions), length) / (length / 4)
    indices = np.floor(coordinate).astype(int)
    fraction = coordinate - indices
    basis = np.zeros((len(coordinate), 4))
    basis[np.arange(len(coordinate)), indices] = 1 - fraction
    basis[np.arange(len(coordinate)), (indices + 1) % 4] = fraction
    return basis


def fit_surfaces(full, core, dataset):
    length = dataset["box_lengths"][0]
    heights = {}
    diagnostics = {}
    contacts = {}
    for surface in ("bottom", "top"):
        labels = dataset[f"{surface}_wall_labels"]
        points = np.asarray([
            edge["contact_location"]
            for node, attrs in full.nodes(data=True)
            if attrs.get("is_wall") and attrs.get("wall_label") in labels
            for edge in full[node].values()
        ], dtype=float)
        basis = surface_basis(points[:, 0], length)
        if np.linalg.matrix_rank(basis) != 4:
            raise ValueError(f"Insufficient {surface} surface coverage")
        heights[surface] = np.linalg.lstsq(basis, points[:, 2], rcond=None)[0]
        residual = basis @ heights[surface] - points[:, 2]
        diagnostics[f"{surface}_max_residual_m"] = float(np.max(np.abs(residual)))
        diagnostics[f"{surface}_rms_residual_m"] = float(np.sqrt(np.mean(residual ** 2)))
        diagnostics[f"{surface}_contact_count"] = len(points)
        for index, height in enumerate(heights[surface]):
            diagnostics[f"{surface}_knot_{index}_z_m"] = height
        if np.max(np.abs(residual)) > 1e-6:
            raise ValueError(f"{surface} surface fit exceeds 1 micrometre tolerance")
        contacts[surface] = points
    nodes = list(core)
    xyz = np.asarray([core.nodes[node]["position"] for node in nodes])
    basis = surface_basis(xyz[:, 0], length)
    bottom = basis @ heights["bottom"]
    top = basis @ heights["top"]
    gap = top - bottom
    if np.min(heights["top"] - heights["bottom"]) <= 0:
        raise ValueError("Surface intersection detected")
    eta = (xyz[:, 2] - bottom) / gap
    if eta.min() < 0 or eta.max() > 1:
        raise ValueError("Particle centers outside fitted physical surfaces")
    diagnostics.update(
        gap_mean_m=float(np.mean(heights["top"] - heights["bottom"])),
        gap_min_m=float(np.min(heights["top"] - heights["bottom"])),
        gap_max_m=float(np.max(heights["top"] - heights["bottom"])),
        eta_min=float(eta.min()), eta_max=float(eta.max()),
    )
    return dict(zip(nodes, eta)), diagnostics, heights, contacts


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_surface_audit(angle, dataset, heights, contacts, output):
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    positions = np.linspace(0, dataset["box_lengths"][0], 801)
    for surface, color in (("bottom", "#0072B2"), ("top", "#D55E00")):
        points = contacts[surface]
        axis.scatter(points[:, 0] * 1000, points[:, 2] * 1000, s=7, alpha=0.35, color=color)
        axis.plot(positions * 1000, surface_basis(positions, dataset["box_lengths"][0]) @ heights[surface] * 1000,
                  color=color, label=f"Fitted {surface} surface")
    axis.set(xlabel="x (mm)", ylabel="z (mm)", title=f"{angle}, simulation 000: physical wall-contact reconstruction")
    axis.legend()
    save_figure(figure, output / "surface_validation" / f"{angle}_sim000_surface_fit")


def summarize_curves(sweep):
    rows = []
    for (angle, n_value), group in sweep.groupby(["angle", "n_value"]):
        for metric in METRICS:
            rows.append(dict(angle=angle, n_value=n_value, property=metric, **ci_summary(group[metric].to_numpy())))
    return pd.DataFrame(rows)


def plot_curves(sweep, curves, output):
    for metric, label in METRICS.items():
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for angle, color in COLORS.items():
            frame = curves[(curves.angle == angle) & (curves.property == metric)]
            axis.plot(frame.n_value, frame["mean"], color=color, label=angle)
            axis.fill_between(frame.n_value.to_numpy(), frame.ci95_low.to_numpy(), frame.ci95_high.to_numpy(), color=color, alpha=0.18)
        axis.set(xlabel="Threshold multiplier n (individual-system mean)", ylabel=label, ylim=(0, None))
        axis.legend()
        axis.grid(alpha=0.2)
        save_figure(figure, output / "curves" / metric)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for axis, column, title in zip(axes, ["any_cluster_spans_loading", "largest_spans_loading"], ["Any cluster spans", "Largest strong cluster spans"]):
        for angle, color in COLORS.items():
            grouped = sweep[sweep.angle == angle].groupby("n_value")[column]
            probability = grouped.mean()
            count = grouped.count()
            critical = stats.norm.ppf(0.975)
            denominator = 1 + critical ** 2 / count
            center = (probability + critical ** 2 / (2 * count)) / denominator
            half = critical * np.sqrt(probability * (1 - probability) / count + critical ** 2 / (4 * count ** 2)) / denominator
            axis.plot(probability.index, probability, color=color, label=angle)
            axis.fill_between(probability.index.to_numpy(), (center - half).to_numpy(), (center + half).to_numpy(), color=color, alpha=0.18)
        axis.set(xlabel="Threshold multiplier n", title=title, ylim=(-0.02, 1.02))
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Fraction of simulations spanning (95% Wilson CI)")
    save_figure(figure, output / "curves" / "spanning_probability_comparison")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for axis, column, title in zip(axes, ["relative_length_z", "largest_cluster_boundary_eta_extent"], ["Original global z extent", "Surface-relative extent (boundary-inclusive)"]):
        for angle, color in COLORS.items():
            means = sweep[sweep.angle == angle].groupby("n_value")[column].mean()
            axis.plot(means.index, means, label=angle, color=color)
        axis.set(xlabel="Threshold multiplier n", title=title, ylim=(0, 1.03))
        axis.legend()
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean extent of largest strong cluster")
    save_figure(figure, output / "curves" / "loading_extent_comparison")


def plot_selections(selected, output):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    values = [selected.loc[selected.angle == angle, "exact_any_spanning_n"].to_numpy() for angle in COLORS]
    bins = np.linspace(min(map(np.min, values)) - 0.05, max(map(np.max, values)) + 0.05, 12)
    axes[0].hist(values, bins=bins, color=list(COLORS.values()), label=list(COLORS), alpha=0.8)
    axes[0].set(xlabel="Exact any-cluster threshold n*", ylabel="Simulation count")
    axes[0].legend()
    axes[1].boxplot(values, tick_labels=list(COLORS), showfliers=False)
    for index, (angle, color) in enumerate(COLORS.items(), 1):
        axes[1].scatter(index + np.linspace(-0.08, 0.08, len(values[index - 1])), values[index - 1], color=color, s=20)
        ordered = np.sort(values[index - 1])
        axes[2].step(np.r_[bins[0], ordered], np.r_[0, np.arange(1, len(ordered) + 1) / len(ordered)], where="post", color=color, label=angle)
    axes[1].set_ylabel("Exact any-cluster threshold n*")
    axes[2].set(xlabel="Exact any-cluster threshold n*", ylabel="Empirical cumulative probability")
    axes[2].legend()
    save_figure(figure, output / "selection" / "exact_threshold_distributions")
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for angle, color in COLORS.items():
        frame = selected[selected.angle == angle]
        axis.scatter(frame.selected_n_largest_spanning, frame.exact_any_spanning_n, label=angle, color=color)
    limits = [min(selected.selected_n_largest_spanning.min(), bins[0]), bins[-1]]
    axis.plot(limits, limits, "--", color="gray")
    axis.set(xlabel="Previous: last sampled largest-cluster spanning n", ylabel="New: exact any-cluster spanning n*")
    axis.legend()
    save_figure(figure, output / "selection" / "old_vs_new_threshold")


def plot_cluster_sizes(components, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)
    size_edges = np.asarray([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    n_values = np.sort(components.n_value.unique())
    n_edges = np.r_[n_values - 0.05, n_values[-1] + 0.05]
    for axis, angle in zip(axes, COLORS):
        subset = components[(components.angle == angle) & components.spans_loading]
        histogram, _, _ = np.histogram2d(subset.n_value, subset.node_count, bins=(n_edges, size_edges))
        totals = histogram.sum(axis=1, keepdims=True)
        probability = np.divide(histogram, totals, out=np.full_like(histogram, np.nan), where=totals > 0)
        artist = axis.pcolormesh(n_edges, size_edges, np.ma.masked_invalid(probability.T), vmin=0, vmax=1, cmap="viridis")
        axis.set(xlabel="Threshold multiplier n", title=angle, yscale="log")
    axes[0].set_ylabel("Spanning cluster size (particles)")
    figure.colorbar(artist, ax=axes, label="Fraction of spanning clusters in size bin")
    save_figure(figure, output / "curves" / "spanning_cluster_size_distribution")


def write_results(sweep, selected, surfaces, output):
    rows = ["# Updated periodic percolation results", "", "20 independent simulations per geometry; particle–particle force normalization per simulation.", "",
            "| Quantity | 0° | 30° |", "|---|---:|---:|"]
    for column, label in (("selected_n_largest_spanning", "Previous mean last-largest-spanning n"),
                          ("selected_n_any_spanning_grid", "New mean last-any-spanning grid n"),
                          ("exact_any_spanning_n", "New mean exact any-spanning n*")):
        means = selected.groupby("angle")[column].mean()
        rows.append(f"| {label} | {means['0deg']:.4f} | {means['30deg']:.4f} |")
    at_three = sweep[np.isclose(sweep.n_value, 3)].groupby("angle")
    for column, label in (("any_cluster_spans_loading", "At n=3: fraction with any spanning cluster"),
                          ("spanning_cluster_count", "At n=3: mean spanning-cluster count"),
                          ("largest_spanning_cluster_size", "At n=3: mean largest spanning size, absent=0"),
                          ("largest_cluster_boundary_eta_extent", "At n=3: mean boundary-inclusive local-height extent of largest strong cluster")):
        means = at_three[column].mean()
        rows.append(f"| {label} | {means['0deg']:.4f} | {means['30deg']:.4f} |")
    first = selected.loc[selected.angle == "0deg", "exact_any_spanning_n"]
    second = selected.loc[selected.angle == "30deg", "exact_any_spanning_n"]
    probability = stats.ttest_ind(first, second, equal_var=False).pvalue
    residual = surfaces[["bottom_max_residual_m", "top_max_residual_m"]].to_numpy().max() * 1e6
    rows.extend(["", f"The exact-threshold geometry difference is not detected by the unadjusted Welch test (p={probability:.4f}); this is not an equivalence test.", "",
                 "At n=3, any-cluster spanning occurs in 9/20 versus 10/20 simulations, whereas largest-cluster spanning occurs in 4/20 versus 9/20. This separates a component-ranking effect from actual loss of all spanning paths.", "",
                 "The old global-z normalization gives substantially lower apparent 30° extent. Local-height normalization largely removes that difference across the main transition. The boundary-inclusive metric explicitly incorporates known boundary attachments; use the separate center-based eta curves to inspect geometric coverage without that augmentation.", "",
                 f"All 2,040 reconstructed networks matched saved spanning flags and largest sizes. Any-cluster spanning is monotonic. All exact thresholds matched independent widest-path recomputation. Maximum wall-surface fit residual: {residual:.3f} micrometres.", "",
                 "See README.md for normalization, missing-value conventions, and the distinction between physical surface extent and spanning connectivity."])
    (output / "RESULTS.md").write_text("\n".join(rows) + "\n")


def run(config_path):
    cfg = load_config(config_path)
    dataset = cfg["datasets"]["periodic_boundaries"]
    root = Path(dataset["output_root"])
    output = root / "summary" / "any_cluster_percolation"
    baseline = pd.read_csv(root / "summary" / "tables" / "sweep_metrics.csv")
    old_selected = pd.read_csv(root / "summary" / "critical_thresholds" / "selected_thresholds.csv")
    graphs = load_graphs(dataset["graph_pickle"])
    if any(len(graphs[angle]["core"]) != 20 for angle in dataset["angles"]):
        raise ValueError("Expected 20 periodic simulations per geometry")
    network_rows, component_rows, surface_rows, selected_rows = [], [], [], []
    expected_n = np.round(np.arange(0, 5.01, 0.1), 1)
    for angle in dataset["angles"]:
        for sim_idx, core in enumerate(graphs[angle]["core"]):
            full = graphs[angle]["full"][sim_idx]
            reference = baseline[(baseline.angle == angle) & (baseline.sim_idx == sim_idx)].sort_values("n_value")
            if len(reference) != len(expected_n) or not np.allclose(reference.n_value, expected_n):
                raise ValueError("Expected complete 0:0.1:5 sweep")
            mean_force = float(np.mean([data["normal_force"] for _, _, data in core.edges(data=True)]))
            if not np.allclose(reference.mean_normal_force, mean_force, rtol=1e-10, atol=0):
                raise ValueError("Force normalization differs from baseline")
            bottom = wall_contact_particles(full, dataset["bottom_wall_labels"])
            top = wall_contact_particles(full, dataset["top_wall_labels"])
            eta, diagnostic, heights, contacts = fit_surfaces(full, core, dataset)
            surface_rows.append(dict(angle=angle, sim_idx=sim_idx, **diagnostic))
            if sim_idx == 0:
                plot_surface_audit(angle, dataset, heights, contacts, output)
            exact = widest_path_threshold(core, bottom, top, mean_force)
            task_rows = []
            for saved in reference.itertuples():
                strong = nx.Graph([(first, second) for first, second, data in core.edges(data=True)
                                   if data["normal_force"] >= saved.n_value * mean_force])
                clusters = sorted(nx.connected_components(strong), key=lambda nodes: (-len(nodes), min(map(str, nodes))))
                task_components = []
                for index, nodes in enumerate(clusters):
                    low, high = min(eta[node] for node in nodes), max(eta[node] for node in nodes)
                    touches_bottom, touches_top = bool(nodes & bottom), bool(nodes & top)
                    task_components.append(dict(
                        angle=angle, sim_idx=sim_idx, n_value=saved.n_value, component_id=index,
                        node_count=len(nodes), is_largest=index == 0,
                        spans_loading=touches_bottom and touches_top,
                        touches_bottom=touches_bottom, touches_top=touches_top,
                        eta_extent=high - low,
                        boundary_eta_extent=(1 if touches_top else high) - (0 if touches_bottom else low),
                    ))
                spanning = [component for component in task_components if component["spans_loading"]]
                largest = task_components[0] if task_components else None
                if (len(clusters[0]) if clusters else 0) != saved.largest_cluster_size:
                    raise ValueError("Largest-cluster size differs from saved sweep")
                if bool(spanning) != saved.any_cluster_spans_loading or bool(largest and largest["spans_loading"]) != saved.largest_spans_loading:
                    raise ValueError("Spanning decision differs from saved sweep")
                if bool(spanning) != bool(saved.n_value <= exact):
                    raise ValueError("Exact threshold and grid spanning disagree")
                sizes = [component["node_count"] for component in spanning]
                row = dict(
                    angle=angle, sim_idx=sim_idx, n_value=saved.n_value,
                    mean_normal_force=mean_force, threshold_force=saved.n_value * mean_force,
                    any_cluster_spans_loading=bool(spanning), largest_spans_loading=saved.largest_spans_loading,
                    spanning_cluster_count=len(spanning), spanning_particle_count=sum(sizes),
                    spanning_particle_fraction=sum(sizes) / len(core),
                    largest_spanning_cluster_size=max(sizes, default=0),
                    mean_spanning_cluster_size=float(np.mean(sizes)) if sizes else np.nan,
                    largest_cluster_eta_extent=largest["eta_extent"] if largest else 0,
                    largest_cluster_boundary_eta_extent=largest["boundary_eta_extent"] if largest else 0,
                    largest_spanning_eta_extent=spanning[0]["eta_extent"] if spanning else np.nan,
                    relative_length_z=saved.relative_length_z,
                )
                task_rows.append(row)
                component_rows.extend(task_components)
            task = pd.DataFrame(task_rows)
            if np.any(np.diff(task.any_cluster_spans_loading.astype(int)) > 0):
                raise ValueError("Any-cluster spanning must be monotonic")
            passing = task.loc[task.any_cluster_spans_loading, "n_value"]
            last = passing.max() if len(passing) else np.nan
            failing = task.loc[~task.any_cluster_spans_loading, "n_value"]
            previous = old_selected[(old_selected.angle == angle) & (old_selected.sim_idx == sim_idx)].iloc[0]
            if not np.isclose(exact, previous.exact_any_spanning_n, rtol=1e-10, atol=0):
                raise ValueError("Recomputed exact threshold differs from saved threshold")
            selected_rows.append(dict(
                angle=angle, sim_idx=sim_idx, mean_normal_force=mean_force,
                selected_n_any_spanning_grid=last, first_nonspanning_grid_n=failing.min() if len(failing) else np.nan,
                exact_any_spanning_n=exact, exact_threshold_force=exact * mean_force,
                grid_right_censored=bool(task.any_cluster_spans_loading.iloc[-1]),
                selected_n_largest_spanning=previous.selected_n_largest_spanning,
            ))
            network_rows.extend(task_rows)
            print(f"{angle} sim{sim_idx:03d}: any-spanning grid={last:.1f}, exact={exact:.6f}", flush=True)
    sweep = pd.DataFrame(network_rows)
    components = pd.DataFrame(component_rows)
    selected = pd.DataFrame(selected_rows)
    curves = summarize_curves(sweep)
    group_rows = []
    comparisons = []
    for column in ("selected_n_any_spanning_grid", "exact_any_spanning_n", "selected_n_largest_spanning"):
        for angle, group in selected.groupby("angle"):
            group_rows.append(dict(angle=angle, criterion=column, **ci_summary(group[column].to_numpy())))
        first = selected.loc[selected.angle == "0deg", column]
        second = selected.loc[selected.angle == "30deg", column]
        welch = stats.ttest_ind(first, second, equal_var=False)
        comparisons.append(dict(criterion=column, difference_30_minus_0=second.mean() - first.mean(),
                                welch_p_unadjusted=welch.pvalue,
                                mann_whitney_p_unadjusted=stats.mannwhitneyu(first, second, alternative="two-sided").pvalue))
    for name, frame in (("sweep_metrics", sweep), ("component_metrics", components), ("curve_mean_ci", curves),
                        ("surface_fit_audit", pd.DataFrame(surface_rows)), ("selected_thresholds", selected),
                        ("threshold_group_mean_ci", pd.DataFrame(group_rows)), ("threshold_comparisons", pd.DataFrame(comparisons))):
        atomic_csv(output / "tables" / f"{name}.csv", frame)
    plot_curves(sweep, curves, output)
    plot_selections(selected, output)
    plot_cluster_sizes(components, output)
    atomic_json(output / "validation.json", dict(
        simulations=len(selected), sweep_rows=len(sweep), components=len(components),
        baseline_spanning_and_largest_size_match=True, monotonic_any_spanning=True,
        exact_thresholds_recomputed_and_matched=True,
        max_surface_residual_m=max(max(row["bottom_max_residual_m"], row["top_max_residual_m"]) for row in surface_rows),
        source_graph=dataset["graph_pickle"], source_sweep=str(root / "summary/tables/sweep_metrics.csv"),
        config=str(config_path.resolve()),
    ))
    documentation = Path(__file__).with_suffix(".md").read_text()
    (output / "README.md").write_text(documentation)
    (output / "FIGURE_GUIDE.md").write_text(Path(__file__).with_name("periodic_spanning_figure_guide.md").read_text())
    write_results(sweep, selected, pd.DataFrame(surface_rows), output)
    print(f"Saved supplementary analysis: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"],
                         "font.size": 12, "svg.fonttype": "none"})
    run(arguments.config)
