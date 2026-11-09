#!/usr/bin/env python3
"""Per-system correlations between betweenness and contact force, and their trend with angle.

Three measures, each computed within every system (systems are the replicates):
  particle: Spearman rho between node betweenness and the particle's largest
            particle-particle normal force (the continuous quantity behind the
            high-force label, so no threshold is involved);
  label:    AUC = P(betweenness of a random high-force particle > that of a random
            non-high-force particle), ties counted half; 0.5 = no relationship;
  contact:  Spearman rho between edge betweenness (computed on the same contact
            network) and the contact's normal force.
Location-controlled versions remove a cubic surface in (r, z_eff) from the ranks of
both variables (z_eff = boundary-following height, as in the r-z maps; contacts use
their midpoints) and correlate the residuals (a partial rank correlation; for the
label it is a partial point-biserial correlation, so its null is 0, not 0.5).
Trend with angle: least-squares slope of the per-system values on angle.

Dataset differences (BC_DATASET, see core_only_common.py):
  stress datasets: the particle measure uses compressive hydrostatic stress (-stress_hydro,
      larger = more compressed) instead of contact force, and there is no contact measure;
      unlabeled particles (no stress) stay in the graph but are left out of the measures.
  location control: final load r and z_eff (cubic surface); experiment r and height (cubic
      surface in their ranks); periodic boundaries height only (cubic in its rank; x, y periodic).
  Mann-Whitney / Spearman p-values are reported per system (useful for the single-sample
      experiment; particles are not independent, so treat them as descriptive).
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
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from core_only_common import (CONTACTS_CSV, DATASET, DATASET_NAME, IS_STRESS, OUTPUT_ROOT, PARTICLES_CSV,
                              angle_key, force_threshold, load_contacts, load_particles, mean_ci,
                              sorted_angles, summarize)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ManuscriptVisualization"))
from plot_final_load_betweenness_rz_bins import SPLIT_R, z_eff_from_rz  # noqa: E402

OUT = OUTPUT_ROOT / "05_correlation"
MEASURES = {
    "rho_particle": ("Particle: betweenness vs largest contact force", "Spearman rho", 0.0),
    "auc_label": ("Label: high-force vs non-high-force betweenness", "AUC", 0.5),
    "rho_contact": ("Contact: edge betweenness vs normal force", "Spearman rho", 0.0),
}
PARTIAL = {
    "partial_particle": ("Particle, location-controlled", "Partial rank correlation"),
    "partial_label": ("Label, location-controlled", "Partial point-biserial correlation"),
    "partial_contact": ("Contact, location-controlled", "Partial rank correlation"),
}
if IS_STRESS:
    MEASURES["rho_particle"] = ("Particle: betweenness vs compressive stress (-hydrostatic)", "Spearman rho", 0.0)
    del MEASURES["rho_contact"], PARTIAL["partial_contact"]
LOCATION = DATASET["location"]


def location(points, center, z_min, angle):
    """(radial, height) location; radial is None for periodic boundaries (height only)."""
    shifted = points - center
    r = np.hypot(shifted[:, 0], shifted[:, 1])
    z = shifted[:, 2] - z_min
    if LOCATION == "sim_zeff":
        return r, z_eff_from_rz(r, z, math.tan(math.radians(angle_key(angle))), SPLIT_R)
    if LOCATION == "r_z":
        return r, z
    return None, z


def cubic_design(r, z_eff):
    v = pd.Series(z_eff).rank(pct=True).to_numpy()
    if r is None:
        return np.column_stack([v ** j for j in range(4)])
    u = pd.Series(r).rank(pct=True).to_numpy()
    return np.column_stack([u ** i * v ** j for i in range(4) for j in range(4 - i)])


def partial_corr(a, b, design):
    """Correlation of a and b after removing the design's least-squares fit from each."""
    residuals = [values - design @ np.linalg.lstsq(design, values, rcond=None)[0] for values in (a, b)]
    return float(np.corrcoef(*residuals)[0, 1])


def ranks(values):
    return pd.Series(values).rank().to_numpy(float)


def system_measures(particles, contacts, threshold):
    rows, edge_tables = [], []
    contact_groups = dict(tuple(contacts.groupby(["angle", "sim_idx"])))
    for (angle, sim), sub in particles.groupby(["angle", "sim_idx"]):
        sub = sub.set_index("node_id")
        edges = contact_groups[angle, sim].copy()
        graph = nx.Graph()
        graph.add_nodes_from(sub.index)
        graph.add_edges_from(zip(edges.edge_u, edges.edge_v))
        recomputed = nx.betweenness_centrality(graph, normalized=True)
        if not np.allclose([recomputed[n] for n in sub.index], sub.betweenness, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{angle} sim {sim}: betweenness from contacts differs from stored values")
        edge_b = nx.edge_betweenness_centrality(graph, normalized=True)
        edges["edge_betweenness"] = [edge_b.get((u, v), edge_b.get((v, u))) for u, v in
                                     zip(edges.edge_u, edges.edge_v)]

        incident = pd.concat([edges[["edge_u", "normal_force"]].rename(columns={"edge_u": "node_id"}),
                              edges[["edge_v", "normal_force"]].rename(columns={"edge_v": "node_id"})])
        sub["max_contact_force"] = incident.groupby("node_id").normal_force.max().reindex(sub.index)
        if not IS_STRESS:
            is_high_all = (sub.group == "high_force").to_numpy()
            if not np.array_equal(is_high_all, (sub.max_contact_force >= threshold).to_numpy()):
                raise ValueError(f"{angle} sim {sim}: labels differ from max contact force >= threshold")

        positions = sub[["x", "y", "z"]].to_numpy(float)
        center = positions.mean(axis=0)
        z_min = (positions - center)[:, 2].min()
        r_p, zeff_p = location(positions, center, z_min, angle)
        midpoints = 0.5 * (sub.loc[edges.edge_u, ["x", "y", "z"]].to_numpy(float)
                           + sub.loc[edges.edge_v, ["x", "y", "z"]].to_numpy(float))
        r_e, zeff_e = location(midpoints, center, z_min, angle)
        edges["midpoint_r"] = np.nan if r_e is None else r_e
        edges["midpoint_z_eff"] = zeff_e

        labeled = sub.group.isin(["high_force", "non_high_force"]).to_numpy()
        lab = sub[labeled]
        is_high = (lab.group == "high_force").to_numpy()
        b = lab.betweenness.to_numpy(float)
        design_p = cubic_design(None if r_p is None else r_p[labeled], zeff_p[labeled])
        if IS_STRESS:  # compressive stress, larger = more compressed; all labeled particles
            score = -lab.stress_hydro.to_numpy(float)
            usable = np.ones(len(lab), bool)
        else:  # largest particle-particle contact force; particles with contacts
            score = lab.max_contact_force.to_numpy(float)
            usable = np.isfinite(score)
        test = stats.mannwhitneyu(b[is_high], b[~is_high])
        rho = stats.spearmanr(b[usable], score[usable])
        row = {
            "angle": angle, "sim_idx": sim, "particles": len(lab), "contacts": len(edges),
            "high_force_particles": int(is_high.sum()),
            "rho_particle": rho.statistic, "rho_particle_p": rho.pvalue,
            "auc_label": test.statistic / (is_high.sum() * (~is_high).sum()), "auc_label_p": test.pvalue,
            "partial_particle": partial_corr(ranks(b[usable]), ranks(score[usable]), design_p[usable]),
            "partial_label": partial_corr(ranks(b), is_high.astype(float), design_p),
        }
        if not IS_STRESS:
            row["rho_contact"] = stats.spearmanr(edges.edge_betweenness, edges.normal_force).statistic
            row["partial_contact"] = partial_corr(ranks(edges.edge_betweenness), ranks(edges.normal_force),
                                                  cubic_design(r_e, zeff_e))
        rows.append(row)
        edge_tables.append(edges)
    return pd.DataFrame(rows), pd.concat(edge_tables, ignore_index=True)


def trend_tests(systems, columns):
    rows = []
    x = systems.angle.map(angle_key).to_numpy(float)
    for column in columns:
        if len(np.unique(x)) < 2 or len(x) < 3:  # e.g. one experiment sample per angle
            rows.append({"measure": column, "slope_per_15deg": np.nan, "slope_ci_low": np.nan,
                         "slope_ci_high": np.nan, "p_value": np.nan, "systems": len(x)})
            continue
        fit = stats.linregress(x, systems[column].to_numpy(float))
        half = stats.t.ppf(0.975, len(x) - 2) * fit.stderr
        rows.append({"measure": column, "slope_per_15deg": 15 * fit.slope,
                     "slope_ci_low": 15 * (fit.slope - half), "slope_ci_high": 15 * (fit.slope + half),
                     "p_value": fit.pvalue, "systems": len(x)})
    return pd.DataFrame(rows)


def plot(systems, trends, angles, path):
    panels = [list(MEASURES.items()), [(k, (t, y, 0.0)) for k, (t, y) in PARTIAL.items()]]
    trend = trends.set_index("measure")
    rng = np.random.default_rng(0)
    x = np.arange(len(angles))
    n_cols = len(MEASURES)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8.4), squeeze=False)
    for row, items in enumerate(panels):
        for col, (column, (title, ylabel, null)) in enumerate(items):
            ax = axes[row, col]
            stats_rows = []
            for i, angle in enumerate(angles):
                values = systems[systems.angle == angle][column]
                ax.scatter(i + rng.uniform(-0.06, 0.06, len(values)), values, s=10, color="#7F7F7F",
                           alpha=0.45, linewidths=0)
                stats_rows.append(mean_ci(values))
            stats_rows = np.array(stats_rows)
            ax.errorbar(x, stats_rows[:, 0], yerr=[stats_rows[:, 0] - stats_rows[:, 1],
                                                   stats_rows[:, 2] - stats_rows[:, 0]],
                        color="#D95F02", marker="o", capsize=3, linewidth=1.6)
            ax.axhline(null, color="k", linestyle="--", linewidth=1)
            t = trend.loc[column]
            if np.isfinite(t.slope_per_15deg):
                ax.text(0.02, 0.97, f"slope {t.slope_per_15deg:+.3f} per 15° (p = {t.p_value:.1e})",
                        transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(angles)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
    what = "stress" if IS_STRESS else "force"
    fig.suptitle(f"Betweenness-{what} correlation per system (dots = systems; mean with 95% CI; "
                 f"{DATASET['title']})")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    threshold = None if IS_STRESS else force_threshold()
    particles = load_particles(include_unlabeled=True)
    contacts = load_contacts()
    angles = sorted_angles(particles.angle)
    systems, edges = system_measures(particles, contacts, threshold)

    tables, figures = OUT / "tables", OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    columns = list(MEASURES) + list(PARTIAL)
    systems.to_csv(tables / "correlations_per_system.csv", index=False)
    edges.to_csv(tables / "contacts_edge_betweenness.csv", index=False)
    summary = summarize(systems, ["angle"], columns)
    summary.to_csv(tables / "correlations_summary.csv", index=False)
    trends = trend_tests(systems, columns)
    trends.to_csv(tables / "trend_tests.csv", index=False)
    plot(systems, trends, angles, figures / "correlation_vs_angle.png")

    (OUT / "metadata.json").write_text(json.dumps({
        "dataset": DATASET_NAME, "inputs": [str(PARTICLES_CSV), str(CONTACTS_CSV)], "threshold": threshold,
        "checks": (["labels == largest particle-particle normal force >= threshold"] if not IS_STRESS else [])
                  + ["node betweenness recomputed from contacts equals stored values"],
        "particle_score": "compressive stress -stress_hydro" if IS_STRESS else "largest contact normal force",
        "measures": {k: v[0] for k, v in {**MEASURES, **{k: (t, y) for k, (t, y) in PARTIAL.items()}}.items()},
        "location_control": {"sim_zeff": "cubic surface in percentile ranks of r and z_eff",
                             "r_z": "cubic surface in percentile ranks of r and height",
                             "z": "cubic polynomial in the percentile rank of height (x, y periodic)"}[LOCATION]
                            + " (contact midpoints for contacts), removed from ranked variables; label kept as 0/1",
        "isolated_particles": "excluded from particle measures (no contact force)",
        "trend": "least-squares slope of per-system values on angle, reported per 15 degrees",
        "replicates": "systems; mean and 95% t interval over systems per angle",
    }, indent=2))
    show = summary[["angle"] + [f"{c}_mean" for c in columns]]
    show.columns = ["angle"] + columns
    print(show.round(3).to_string(index=False))
    print(trends.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
