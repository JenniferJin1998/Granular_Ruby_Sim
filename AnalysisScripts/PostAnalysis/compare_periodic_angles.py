#!/usr/bin/env python3
"""Compare 0-degree and 30-degree periodic-boundary graph properties.

Node and edge observations are summarized within each simulation. Statistical
tests are then performed on simulation-level summaries, so a particle or
contact is never treated as an independent experimental replicate. Graph
properties already have one value per simulation and are compared directly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial import distance as spatial_distance


DATE_TAG = "2026-08-03"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "AnalysisResults" / "PeriodicBoudaries" / DATE_TAG / "GraphPipeline"
DEFAULT_OUTPUT = PROJECT_ROOT / "AnalysisResults" / "PeriodicBoudaries" / DATE_TAG / "AngleComparison"
ANGLES = ("0deg", "30deg")

ID_COLUMNS = {
    "node": {"geometry", "sim_idx", "node_id"},
    "edge": {"geometry", "sim_idx", "node1", "node2"},
    "graph": {"geometry", "sim_idx"},
}

# Coordinates identify locations rather than graph/force response properties.
# They can be restored with --include-coordinates.
COORDINATE_COLUMNS = {
    "node": {"x", "y", "z"},
    "edge": {"contact_x", "contact_y", "contact_z"},
    "graph": set(),
}

STAT_NAMES = [
    "n_observations",
    "mean",
    "median",
    "minimum",
    "maximum",
    "standard_deviation",
    "variance",
    "iqr",
    "mad",
    "p10_p90_width",
    "fwhm",
    "coefficient_of_variation",
    "p05",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
    "p99",
    "highest_10_percent_mean",
    "fraction_above_threshold",
    "skewness",
    "kurtosis",
    "mode",
    "peak_density",
    "entropy",
    "bimodality_coefficient",
    "number_of_detected_peaks",
]

PLOT_STATS = ["mean", "standard_deviation", "iqr", "fwhm", "skewness", "kurtosis"]
COLORS = {"0deg": "#2474B5", "30deg": "#D95F02"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def finite_values(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def parse_thresholds(specs: list[str]) -> tuple[dict[str, float], float | None]:
    thresholds: dict[str, float] = {}
    default = None
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid threshold {spec!r}; use PROPERTY=VALUE or default=VALUE")
        key, raw = spec.split("=", 1)
        if key.strip().lower() in {"default", "*"}:
            default = float(raw)
        else:
            thresholds[key.strip()] = float(raw)
    return thresholds, default


def kde_characteristics(values: np.ndarray, grid_size: int = 512) -> dict[str, float | bool]:
    result = {
        "fwhm": np.nan,
        "mode": np.nan,
        "peak_density": np.nan,
        "number_of_detected_peaks": np.nan,
        "multimodal": False,
        "fwhm_reliable": False,
    }
    if values.size < 5 or np.unique(values).size < 3 or np.ptp(values) <= 0:
        return result
    try:
        kde = stats.gaussian_kde(values)
        q01, q99 = np.percentile(values, [1, 99])
        spread = max(q99 - q01, np.std(values, ddof=1), np.finfo(float).eps)
        lo = min(float(values.min()), q01 - 0.15 * spread)
        hi = max(float(values.max()), q99 + 0.15 * spread)
        grid = np.linspace(lo, hi, grid_size)
        density = kde(grid)
        if not np.all(np.isfinite(density)) or density.max() <= 0:
            return result

        peak_idx, _ = find_peaks(
            density,
            prominence=max(0.05 * float(density.max()), np.finfo(float).eps),
            distance=max(2, grid_size // 50),
        )
        if peak_idx.size == 0:
            peak_idx = np.asarray([int(np.argmax(density))])
        dominant = int(peak_idx[np.argmax(density[peak_idx])])
        result.update(
            mode=float(grid[dominant]),
            peak_density=float(density[dominant]),
            number_of_detected_peaks=int(peak_idx.size),
            multimodal=bool(peak_idx.size > 1),
        )

        # FWHM is reported only for a clearly unimodal KDE with crossings on
        # both sides of its dominant peak.
        if peak_idx.size != 1 or dominant == 0 or dominant == grid_size - 1:
            return result
        half = density[dominant] / 2.0
        left_candidates = np.where(density[:dominant] <= half)[0]
        right_candidates = np.where(density[dominant + 1 :] <= half)[0]
        if left_candidates.size == 0 or right_candidates.size == 0:
            return result
        li = int(left_candidates[-1])
        ri = int(dominant + 1 + right_candidates[0])
        left = np.interp(half, density[li : li + 2], grid[li : li + 2])
        # Reverse the descending right segment for np.interp.
        right = np.interp(half, density[ri - 1 : ri + 1][::-1], grid[ri - 1 : ri + 1][::-1])
        width = float(right - left)
        if np.isfinite(width) and width > 0:
            result["fwhm"] = width
            result["fwhm_reliable"] = True
    except (ValueError, np.linalg.LinAlgError):
        pass
    return result


def histogram_entropy(values: np.ndarray) -> float:
    if values.size < 2 or np.ptp(values) <= 0:
        return 0.0 if values.size else np.nan
    edges = np.histogram_bin_edges(values, bins="fd")
    if len(edges) < 2:
        return np.nan
    counts, _ = np.histogram(values, bins=edges)
    probabilities = counts[counts > 0] / counts.sum()
    return float(stats.entropy(probabilities))


def distribution_statistics(values, threshold: float | None) -> dict[str, float | bool]:
    x = finite_values(values)
    output: dict[str, float | bool] = {name: np.nan for name in STAT_NAMES}
    output.update(threshold=np.nan if threshold is None else threshold, multimodal=False, fwhm_reliable=False)
    output["n_observations"] = int(x.size)
    if x.size == 0:
        return output

    q05, q10, q25, q75, q90, q95, q99 = np.percentile(x, [5, 10, 25, 75, 90, 95, 99])
    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    variance = float(np.var(x, ddof=1)) if x.size > 1 else np.nan
    mad = float(np.median(np.abs(x - median)))
    top_cut = float(np.percentile(x, 90))
    top = x[x >= top_cut]
    skew = float(stats.skew(x, bias=False)) if x.size >= 3 and np.ptp(x) > 0 else np.nan
    kurt = float(stats.kurtosis(x, fisher=True, bias=False)) if x.size >= 4 and np.ptp(x) > 0 else np.nan
    pearson_kurtosis = kurt + 3.0 if np.isfinite(kurt) else np.nan
    bc = (skew * skew + 1.0) / pearson_kurtosis if np.isfinite(pearson_kurtosis) and pearson_kurtosis > 0 else np.nan

    output.update(
        mean=mean,
        median=median,
        minimum=float(np.min(x)),
        maximum=float(np.max(x)),
        standard_deviation=std,
        variance=variance,
        iqr=float(q75 - q25),
        mad=mad,
        p10_p90_width=float(q90 - q10),
        coefficient_of_variation=float(std / abs(mean)) if np.isfinite(std) and mean != 0 else np.nan,
        p05=float(q05),
        p10=float(q10),
        p25=float(q25),
        p75=float(q75),
        p90=float(q90),
        p95=float(q95),
        p99=float(q99),
        highest_10_percent_mean=float(np.mean(top)) if top.size else np.nan,
        fraction_above_threshold=float(np.mean(x > threshold)) if threshold is not None else np.nan,
        skewness=skew,
        kurtosis=kurt,
        entropy=histogram_entropy(x),
        bimodality_coefficient=float(bc) if np.isfinite(bc) else np.nan,
    )
    output.update(kde_characteristics(x))
    return output


def numeric_properties(df: pd.DataFrame, level: str, include_coordinates: bool) -> list[str]:
    excluded = set(ID_COLUMNS[level])
    if not include_coordinates:
        excluded |= COORDINATE_COLUMNS[level]
    properties = []
    for column in df.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            properties.append(column)
    return properties


def angle_threshold(prop: str, thresholds: dict[str, float], default: float | None) -> float | None:
    return thresholds.get(prop, default)


def summarize_entity(
    df: pd.DataFrame,
    level: str,
    properties: list[str],
    thresholds: dict[str, float],
    default_threshold: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    simulation_rows = []
    pooled_rows = []
    aggregate_rows = []
    selected = df[df["geometry"].isin(ANGLES)].copy()

    for prop in properties:
        threshold = angle_threshold(prop, thresholds, default_threshold)
        for (angle, sim_idx), group in selected.groupby(["geometry", "sim_idx"], sort=True):
            simulation_rows.append(
                {"entity": level, "property": prop, "geometry": angle, "sim_idx": int(sim_idx),
                 **distribution_statistics(group[prop], threshold)}
            )
        for angle, group in selected.groupby("geometry", sort=True):
            pooled_rows.append(
                {"entity": level, "property": prop, "geometry": angle,
                 **distribution_statistics(group[prop], threshold)}
            )

    sim = pd.DataFrame(simulation_rows)
    pooled = pd.DataFrame(pooled_rows)
    for (prop, angle), group in sim.groupby(["property", "geometry"], sort=True):
        for stat_name in STAT_NAMES:
            vals = finite_values(group[stat_name])
            aggregate_rows.append({
                "entity": level,
                "property": prop,
                "geometry": angle,
                "simulation_statistic": stat_name,
                "n_simulations": int(vals.size),
                "mean_across_simulations": float(np.mean(vals)) if vals.size else np.nan,
                "standard_deviation_across_simulations": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            })
    return sim, pooled, pd.DataFrame(aggregate_rows)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return np.nan
    pooled_df = a.size + b.size - 2
    pooled_var = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / pooled_df
    if pooled_var <= 0:
        return 0.0 if np.mean(a) == np.mean(b) else np.nan
    d = (np.mean(b) - np.mean(a)) / math.sqrt(pooled_var)  # positive means 30deg > 0deg
    correction = 1.0 - 3.0 / (4.0 * pooled_df - 1.0) if pooled_df > 1 else 1.0
    return float(correction * d)


def mann_whitney_safe(a: np.ndarray, b: np.ndarray):
    """Support older SciPy versions that reject two identical constant samples."""
    if not a.size or not b.size:
        return np.nan, np.nan
    if np.ptp(np.concatenate([a, b])) == 0:
        return float(a.size * b.size / 2.0), 1.0
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def replicate_tests(sim_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (entity, prop), prop_df in sim_stats.groupby(["entity", "property"], sort=True):
        for stat_name in STAT_NAMES:
            for geometry_a, geometry_b in combinations(ANGLES, 2):
                a = finite_values(prop_df.loc[prop_df.geometry == geometry_a, stat_name]);b = finite_values(prop_df.loc[prop_df.geometry == geometry_b, stat_name])
                row = {"entity": entity, "property": prop, "simulation_statistic": stat_name,"geometry_a":geometry_a,"geometry_b":geometry_b,"n_simulations_a":int(a.size),"n_simulations_b":int(b.size),"hedges_g_b_minus_a":hedges_g(a,b)}
                if a.size >= 2 and b.size >= 2:
                    t=stats.ttest_ind(a,b,equal_var=False,nan_policy="omit");u_stat,u_p=mann_whitney_safe(a,b);row.update(welch_t=float(t.statistic),welch_p=float(t.pvalue),mann_whitney_u=u_stat,mann_whitney_p=u_p)
                else: row.update(welch_t=np.nan,welch_p=np.nan,mann_whitney_u=np.nan,mann_whitney_p=np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def direct_graph_analysis(graph_df: pd.DataFrame, properties: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = graph_df[graph_df.geometry.isin(ANGLES)][["geometry", "sim_idx", *properties]].copy()
    summary_rows, test_rows = [], []
    for prop in properties:
        for angle in ANGLES:
            x = finite_values(values.loc[values.geometry == angle, prop])
            summary_rows.append({"entity": "graph", "property": prop, "geometry": angle,
                                 **distribution_statistics(x, None)})
        for geometry_a,geometry_b in combinations(ANGLES,2):
            a=finite_values(values.loc[values.geometry==geometry_a,prop]);b=finite_values(values.loc[values.geometry==geometry_b,prop]);t=stats.ttest_ind(a,b,equal_var=False,nan_policy="omit") if a.size>=2 and b.size>=2 else None;u_stat,u_p=mann_whitney_safe(a,b)
            test_rows.append({"entity":"graph","property":prop,"geometry_a":geometry_a,"geometry_b":geometry_b,"n_simulations_a":int(a.size),"n_simulations_b":int(b.size),"welch_t":float(t.statistic) if t else np.nan,"welch_p":float(t.pvalue) if t else np.nan,"mann_whitney_u":u_stat,"mann_whitney_p":u_p,"hedges_g_b_minus_a":hedges_g(a,b)})
    return values, pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def js_distance(a: np.ndarray, b: np.ndarray, bins: int = 128) -> float:
    if not a.size or not b.size:
        return np.nan
    combined = np.concatenate([a, b])
    lo, hi = np.min(combined), np.max(combined)
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    return float(spatial_distance.jensenshannon(pa + 1e-12, pb + 1e-12, base=2.0))


def distribution_distances(df: pd.DataFrame, level: str, properties: list[str]) -> pd.DataFrame:
    rows = []
    for prop in properties:
        for geometry_a,geometry_b in combinations(ANGLES,2):
            a=finite_values(df.loc[df.geometry==geometry_a,prop]);b=finite_values(df.loc[df.geometry==geometry_b,prop]);ks=stats.ks_2samp(a,b) if a.size and b.size else None
            rows.append({"entity":level,"property":prop,"geometry_a":geometry_a,"geometry_b":geometry_b,"n_observations_a":int(a.size),"n_observations_b":int(b.size),"wasserstein_distance":float(stats.wasserstein_distance(a,b)) if a.size and b.size else np.nan,"kolmogorov_smirnov_statistic":float(ks.statistic) if ks else np.nan,"kolmogorov_smirnov_p_descriptive_only":float(ks.pvalue) if ks else np.nan,"jensen_shannon_distance":js_distance(a,b),"energy_distance":float(stats.energy_distance(a,b)) if a.size and b.size else np.nan})
    return pd.DataFrame(rows)


def common_edges(a: np.ndarray, b: np.ndarray, bins: int = 80) -> np.ndarray | None:
    x = np.concatenate([a, b])
    if not x.size or np.ptp(x) <= 0:
        return None
    lo, hi = np.percentile(x, [0.25, 99.75])
    if hi <= lo:
        lo, hi = np.min(x), np.max(x)
    return np.linspace(lo, hi, bins + 1)


def plot_distributions(df: pd.DataFrame, level: str, properties: list[str], out_dir: Path) -> None:
    pooled_dir = out_dir / "pooled_distributions" / level
    individual_dir = out_dir / "individual_simulation_distributions" / level
    pooled_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)
    for prop in properties:
        arrays={angle:finite_values(df.loc[df.geometry==angle,prop]) for angle in ANGLES};nonempty=[x for x in arrays.values() if x.size]
        edges=common_edges(nonempty[0],np.concatenate(nonempty[1:]) if len(nonempty)>1 else nonempty[0]) if nonempty else None
        if edges is None:
            continue
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for angle,x in arrays.items():
            ax.hist(x, bins=edges, density=True, histtype="step", linewidth=1.8,
                    color=COLORS[angle], label=angle)
        ax.set(title=f"Pooled {level}: {prop}", xlabel=prop, ylabel="Density")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(pooled_dir / f"{safe_name(prop)}.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(len(ANGLES), 1, figsize=(7.0,3.0*len(ANGLES)), sharex=True, sharey=True,squeeze=False)
        for ax, angle in zip(axes.flat, ANGLES):
            subset = df[df.geometry == angle]
            for _, group in subset.groupby("sim_idx", sort=True):
                x = finite_values(group[prop])
                if x.size:
                    hist, _ = np.histogram(x, bins=edges, density=True)
                    centers = (edges[:-1] + edges[1:]) / 2
                    ax.plot(centers, hist, color=COLORS[angle], alpha=0.28, linewidth=0.7)
            ax.set_ylabel("Density")
            ax.set_title(f"{angle}: individual simulations")
        axes.flat[-1].set_xlabel(prop)
        fig.suptitle(f"Simulation distributions — {level}: {prop}")
        fig.tight_layout()
        fig.savefig(individual_dir / f"{safe_name(prop)}.png", dpi=180)
        plt.close(fig)


def plot_simulation_statistics(sim_stats: pd.DataFrame, level: str, properties: list[str], out_dir: Path) -> None:
    target = out_dir / "simulation_level_statistics" / level
    target.mkdir(parents=True, exist_ok=True)
    for prop in properties:
        data = sim_stats[sim_stats.property == prop]
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
        for ax, stat_name in zip(axes.flat, PLOT_STATS):
            arrays = [finite_values(data.loc[data.geometry == angle, stat_name]) for angle in ANGLES]
            ax.boxplot(arrays, labels=ANGLES, showfliers=False)
            for idx, (angle, vals) in enumerate(zip(ANGLES, arrays), start=1):
                ax.scatter(np.full(vals.size, idx), vals, color=COLORS[angle], s=13, alpha=0.7)
            ax.set_title(stat_name.replace("_", " "))
        fig.suptitle(f"Simulation-level summaries — {level}: {prop}")
        fig.tight_layout()
        fig.savefig(target / f"{safe_name(prop)}.png", dpi=180)
        plt.close(fig)


def plot_distances(distances: pd.DataFrame, out_dir: Path) -> None:
    target = out_dir / "distribution_distances"
    target.mkdir(parents=True, exist_ok=True)
    metrics = ["wasserstein_distance", "kolmogorov_smirnov_statistic", "jensen_shannon_distance", "energy_distance"]
    for (level,geometry_a,geometry_b), group in distances.groupby(["entity","geometry_a","geometry_b"]):
        fig, axes = plt.subplots(2, 2, figsize=(13.0, max(7.0, 0.25 * len(group))))
        y = np.arange(len(group))
        for ax, metric in zip(axes.flat, metrics):
            ax.barh(y, group[metric], color="#5B8DB8")
            ax.set_yticks(y)
            ax.set_yticklabels(group.property if metric in metrics[::2] else [""] * len(y))
            ax.set_title(metric.replace("_", " "))
            ax.invert_yaxis()
        fig.suptitle(f"Full-distribution distances: {level}, {geometry_a} vs {geometry_b}")
        fig.tight_layout()
        fig.savefig(target / f"{level}_{safe_name(geometry_a)}_vs_{safe_name(geometry_b)}_distance_comparison.png", dpi=180)
        plt.close(fig)


def plot_graph_properties(values: pd.DataFrame, properties: list[str], out_dir: Path) -> None:
    target = out_dir / "graph_level_properties"
    target.mkdir(parents=True, exist_ok=True)
    for prop in properties:
        arrays = [finite_values(values.loc[values.geometry == angle, prop]) for angle in ANGLES]
        if not any(x.size for x in arrays):
            continue
        fig, ax = plt.subplots(figsize=(5.4, 4.5))
        ax.boxplot(arrays, labels=ANGLES, showfliers=False)
        for idx, (angle, vals) in enumerate(zip(ANGLES, arrays), start=1):
            ax.scatter(np.full(vals.size, idx), vals, color=COLORS[angle], s=22, alpha=0.75)
        ax.set(title=f"Graph property: {prop}", ylabel=prop)
        fig.tight_layout()
        fig.savefig(target / f"{safe_name(prop)}.png", dpi=180)
        plt.close(fig)


def write_method_notes(path: Path, properties: dict[str, list[str]], thresholds: dict[str, float], default_threshold) -> None:
    notes = {
        "independent_replicate": "simulation",
        "angles": list(ANGLES),
        "properties": properties,
        "thresholds": thresholds,
        "default_threshold": default_threshold,
        "kurtosis": "Fisher excess kurtosis, bias-corrected",
        "mad": "median(abs(x - median(x))); unscaled",
        "entropy": "Shannon entropy of Freedman-Diaconis histogram probabilities",
        "effect_size": "bias-corrected Hedges g; positive means geometry_b > geometry_a",
        "fwhm": "Gaussian KDE; NaN unless a single detected peak and two half-height crossings exist",
        "multimodal": "more than one KDE peak with prominence >= 5% of maximum KDE density",
        "distribution_distance_note": "Distances are descriptive pooled-distribution comparisons; inferential tests use simulation replicates.",
        "coordinates_default": "Excluded unless --include-coordinates is supplied",
    }
    path.write_text(json.dumps(notes, indent=2) + "\n")


def main() -> None:
    global ANGLES,COLORS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", action="append", default=[], metavar="PROPERTY=VALUE",
                        help="Threshold for fraction-above; repeatable. Use default=VALUE for a fallback.")
    parser.add_argument("--include-coordinates", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--geometries", nargs="+", help="Geometry labels to compare; defaults to 0deg 30deg")
    args = parser.parse_args()
    if args.geometries: ANGLES=tuple(args.geometries)
    cmap=plt.get_cmap("tab10" if len(ANGLES)<=10 else "tab20");COLORS={angle:cmap(i%cmap.N) for i,angle in enumerate(ANGLES)}
    thresholds, default_threshold = parse_thresholds(args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    properties = {}
    for level in ("node", "edge", "graph"):
        path = args.input_dir / f"{level}_features.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[level] = pd.read_csv(path, low_memory=False)
        missing_angles = set(ANGLES) - set(tables[level]["geometry"].astype(str))
        if missing_angles:
            raise ValueError(f"{path} is missing angles: {sorted(missing_angles)}")
        properties[level] = numeric_properties(tables[level], level, args.include_coordinates)
        print(f"{level}: {len(properties[level])} numeric properties")

    all_sim, all_pooled, all_aggregate, all_distances = [], [], [], []
    for level in ("node", "edge"):
        sim, pooled, aggregate = summarize_entity(
            tables[level], level, properties[level], thresholds, default_threshold
        )
        all_sim.append(sim)
        all_pooled.append(pooled)
        all_aggregate.append(aggregate)
        all_distances.append(distribution_distances(tables[level], level, properties[level]))

    simulation_stats = pd.concat(all_sim, ignore_index=True)
    pooled_stats = pd.concat(all_pooled, ignore_index=True)
    aggregate_stats = pd.concat(all_aggregate, ignore_index=True)
    distances = pd.concat(all_distances, ignore_index=True)
    tests = replicate_tests(simulation_stats)
    graph_values, graph_summary, graph_tests = direct_graph_analysis(tables["graph"], properties["graph"])

    simulation_stats.to_csv(args.output_dir / "simulation_distribution_statistics.csv", index=False)
    pooled_stats.to_csv(args.output_dir / "pooled_distribution_statistics.csv", index=False)
    aggregate_stats.to_csv(args.output_dir / "simulation_statistic_angle_summary.csv", index=False)
    distances.to_csv(args.output_dir / "pooled_distribution_distances.csv", index=False)
    tests.to_csv(args.output_dir / "simulation_replicate_tests.csv", index=False)
    graph_values.to_csv(args.output_dir / "graph_property_values_by_simulation.csv", index=False)
    graph_summary.to_csv(args.output_dir / "graph_property_angle_summary.csv", index=False)
    graph_tests.to_csv(args.output_dir / "graph_property_tests.csv", index=False)
    write_method_notes(args.output_dir / "analysis_metadata.json", properties, thresholds, default_threshold)

    if not args.skip_plots:
        for level, sim in zip(("node", "edge"), all_sim):
            plot_distributions(tables[level], level, properties[level], args.output_dir)
            plot_simulation_statistics(sim, level, properties[level], args.output_dir)
        plot_distances(distances, args.output_dir)
        plot_graph_properties(graph_values, properties["graph"], args.output_dir)
    print(f"Saved geometry comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
