#!/usr/bin/env python3
"""Simulation-mean comparison of top-bottom betweenness (BC_DATASET=periodic_force_top_bottom only).

Same replicate convention as the other property comparisons
(2_force_cluster_comparison/.../high_vs_non_high and 1_network_property_comparison/feature_properties):
each system gives one mean (and median) per group, and systems are the replicates.

Properties (per particle, core graph):
  top_bottom            top-bottom betweenness (bottom-wall x top-wall pairs only)
  top_bottom_rel        top_bottom / its system's mean (1 = system average; scale-free)
  all_pairs             standard all-pairs betweenness (folder 5)
  all_pairs_rel         all_pairs / its system's mean
Scopes: all particles, and interior particles only (no wall contact), because wall-contact
particles are endpoints of many top-bottom pairs and get no credit for them.

Tests (tables/statistical_comparisons.csv; effect size oriented b minus a):
  high_vs_non_within_<angle>    paired t and Wilcoxon over systems; effect = mean / SD of the paired difference
  <group>_30deg_vs_0deg          Welch t, Mann-Whitney U, Hedges g of the system means (group = all, high_force,
                                 non_high_force)
  high_minus_non_30deg_vs_0deg   same tests on the per-system high minus non difference
Separation (tables/separation_top_bottom_vs_all_pairs*.csv): the per-system measures of step 05 (AUC,
Spearman rho, partial correlations) and step 02 (top-k enrichment) of folder 5 (all pairs) and this
folder (top-bottom), compared with paired t / Wilcoxon over systems within each angle.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from core_only_common import (DATASET_NAME, DATASETS, GROUP_COLORS, GROUP_LABELS, GROUPS, IS_TOP_BOTTOM, OUTPUT_ROOT,
                              PARTICLES_CSV, angle_key, load_particles, mean_ci, sorted_angles)

OUT = OUTPUT_ROOT / "10_simulation_mean_comparison"
ALL_PAIRS_ROOT = DATASETS["periodic_force"]["output_root"]
PROPERTIES = {
    "top_bottom": "Top-bottom betweenness",
    "top_bottom_rel": "Top-bottom betweenness / system mean",
    "all_pairs": "All-pairs betweenness",
    "all_pairs_rel": "All-pairs betweenness / system mean",
}
SCOPES = {"all": "all particles", "interior": "interior particles (no wall contact)"}
STATISTICS = ("mean", "median")
SEPARATION = {
    "auc_label": ("AUC, high-force vs non-high-force", 0.5),
    "rho_particle": ("Spearman rho, particle vs largest contact force", 0.0),
    "rho_contact": ("Spearman rho, edge betweenness vs normal force", 0.0),
    "partial_label": ("Height-controlled partial, label", 0.0),
    "partial_particle": ("Height-controlled partial, particle", 0.0),
    "partial_contact": ("Height-controlled partial, contact", 0.0),
    "enrichment_top10": ("Top-10% enrichment", 1.0),
    "enrichment_top20": ("Top-20% enrichment", 1.0),
}


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def replicate_test(a, b, paired):
    """Same tests and orientation (b minus a) as PeriodicRubyPipeline/pipeline_common.replicate_test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"n_a": len(a), "n_b": len(b), "mean_a": a.mean(), "mean_b": b.mean()}
    if paired:
        t = stats.ttest_rel(b, a)
        w = stats.wilcoxon(b, a)
        delta = b - a
        out.update(test="paired_t", test_statistic=t.statistic, test_p=t.pvalue, nonparametric_test="wilcoxon",
                   nonparametric_statistic=w.statistic, nonparametric_p=w.pvalue,
                   effect_size=delta.mean() / delta.std(ddof=1), effect_size_kind="mean / SD of paired difference")
    else:
        t = stats.ttest_ind(b, a, equal_var=False)
        u = stats.mannwhitneyu(b, a, alternative="two-sided")
        df = len(a) + len(b) - 2
        pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / df
        out.update(test="welch_t", test_statistic=t.statistic, test_p=t.pvalue, nonparametric_test="mann_whitney_u",
                   nonparametric_statistic=u.statistic, nonparametric_p=u.pvalue,
                   effect_size=(b.mean() - a.mean()) / np.sqrt(pooled) * (1 - 3 / (4 * df - 1)),
                   effect_size_kind="Hedges g")
    return out


def particle_properties(particles):
    by_system = particles.groupby(["angle", "sim_idx"])
    particles["top_bottom"] = particles.betweenness
    particles["all_pairs"] = particles.betweenness_all_pairs
    particles["top_bottom_rel"] = particles.top_bottom / by_system.top_bottom.transform("mean")
    particles["all_pairs_rel"] = particles.all_pairs / by_system.all_pairs.transform("mean")
    return particles


def system_statistics(particles):
    """One row per scope, angle, system, group (incl. 'all') and property."""
    rows = []
    for scope in SCOPES:
        frame = particles if scope == "all" else particles[particles.wall_side == "none"]
        for (angle, sim), sub in frame.groupby(["angle", "sim_idx"]):
            for group in ("all",) + GROUPS:
                part = sub if group == "all" else sub[sub.group == group]
                for prop in PROPERTIES:
                    rows.append({"scope": scope, "angle": angle, "sim_idx": sim, "group": group, "property": prop,
                                 "n": len(part), "mean": part[prop].mean(), "median": part[prop].median()})
    return pd.DataFrame(rows)


def comparisons(systems, angles):
    rows = []
    a0, a1 = angles[0], angles[-1]
    for (scope, prop), sub in systems.groupby(["scope", "property"]):
        for statistic in STATISTICS:
            wide = sub.pivot_table(index=["angle", "sim_idx"], columns="group", values=statistic)
            wide["difference"] = wide.high_force - wide.non_high_force
            key = {"scope": scope, "property": prop, "statistic": statistic}
            for angle in angles:
                w = wide.loc[angle]
                rows.append({**key, "comparison": f"high_vs_non_within_{angle}", "angle": angle,
                             **replicate_test(w.non_high_force, w.high_force, paired=True)})
            for column, name in (("all", "all"), ("high_force", "high_force"), ("non_high_force", "non_high_force"),
                                 ("difference", "high_minus_non")):
                if name == "all" and scope == "all" and statistic == "mean" and prop.endswith("_rel"):
                    continue  # identically 1 by construction
                rows.append({**key, "comparison": f"{name}_{a1}_vs_{a0}", "angle": "between_geometries",
                             "geometry_a": a0, "geometry_b": a1,
                             **replicate_test(wide.loc[a0][column], wide.loc[a1][column], paired=False)})
    return pd.DataFrame(rows)


def summary_table(systems):
    rows = []
    for keys, sub in systems.groupby(["scope", "property", "angle", "group"]):
        row = dict(zip(["scope", "property", "angle", "group"], keys), systems=len(sub))
        for statistic in STATISTICS:
            row[f"{statistic}_mean"], row[f"{statistic}_ci_low"], row[f"{statistic}_ci_high"] = mean_ci(sub[statistic])
        rows.append(row)
    return pd.DataFrame(rows)


def p_text(p):
    return "p < 1e-4" if p < 1e-4 else f"p = {p:.2g}"


def violin_panel(ax, sub, angles, statistic, prop, tests, rng):
    cats = [(angle, group) for angle in angles for group in ("non_high_force", "high_force")]
    for i, (angle, group) in enumerate(cats):
        values = finite(sub.loc[(sub.angle == angle) & (sub.group == group), statistic])
        parts = ax.violinplot(values, [i], showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(GROUP_COLORS[group])
            body.set_alpha(0.25)
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(values)), values, s=15, color=GROUP_COLORS[group], alpha=0.8)
    ax.set_xticks(range(len(cats)), [f"{a} {'non' if g == 'non_high_force' else 'high'}" for a, g in cats],
                  rotation=45, ha="right")
    ax.set_ylabel(f"simulation {statistic}")
    ax.set_title(PROPERTIES[prop], fontsize=10)
    lines = []
    for angle in angles:
        t = tests[tests.comparison == f"high_vs_non_within_{angle}"].iloc[0]
        lines.append(f"{angle} high vs non: d = {t.effect_size:.1f}, Wilcoxon {p_text(t.nonparametric_p)}")
    for name in ("all", "high_minus_non"):
        t = tests[tests.comparison == f"{name}_{angles[-1]}_vs_{angles[0]}"]
        if t.empty:
            continue
        t = t.iloc[0]
        label = "all particles" if name == "all" else "high - non"
        lines.append(f"{angles[-1]} vs {angles[0]}, {label}: g = {t.effect_size:+.2f}, MWU {p_text(t.nonparametric_p)}")
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=6.5,
            bbox=dict(facecolor="white", alpha=0.75, linewidth=0))
    ax.margins(y=0.35)


def plot_simulation_means(systems, tests, angles, root, seed):
    paths = []
    for scope in SCOPES:
        for statistic in STATISTICS:
            target = root / scope
            target.mkdir(parents=True, exist_ok=True)
            # One figure per property (same layout as the existing simulation_mean_boxplots/node/*.png) ...
            for prop in PROPERTIES:
                fig, ax = plt.subplots(figsize=(7, 4.6))
                sub = systems[(systems.scope == scope) & (systems.property == prop)]
                t = tests[(tests.scope == scope) & (tests.property == prop) & (tests.statistic == statistic)]
                violin_panel(ax, sub, angles, statistic, prop, t, np.random.default_rng(seed))
                ax.set_title(f"node: {PROPERTIES[prop]} ({statistic}, {SCOPES[scope]})", fontsize=10)
                fig.tight_layout()
                path = target / f"{prop}__{statistic}.png"
                fig.savefig(path, dpi=180)
                plt.close(fig)
                paths.append(path)
            # ... and a 2 x 2 overview: raw and system-relative, top-bottom vs all pairs.
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for ax, prop in zip(axes.flat, ["top_bottom", "all_pairs", "top_bottom_rel", "all_pairs_rel"]):
                sub = systems[(systems.scope == scope) & (systems.property == prop)]
                t = tests[(tests.scope == scope) & (tests.property == prop) & (tests.statistic == statistic)]
                violin_panel(ax, sub, angles, statistic, prop, t, np.random.default_rng(seed))
            fig.suptitle(f"Simulation {statistic} per group: top-bottom vs all-pairs betweenness "
                         f"({SCOPES[scope]}; dots = systems)")
            fig.tight_layout()
            path = root / f"overview__{scope}__{statistic}.png"
            fig.savefig(path, dpi=200)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
            paths.append(path)
    return paths


def separation_per_system():
    frames = []
    for version, root in (("all_pairs", ALL_PAIRS_ROOT), ("top_bottom", OUTPUT_ROOT)):
        corr = pd.read_csv(root / "05_correlation" / "tables" / "correlations_per_system.csv")
        topk = pd.read_csv(root / "02_global_betweenness_rank" / "tables" / "top_k_per_system.csv")
        for k in (10, 20):
            part = topk[np.isclose(topk.top_percent, k)][["angle", "sim_idx", "enrichment_high_force"]]
            corr = corr.merge(part.rename(columns={"enrichment_high_force": f"enrichment_top{k}"}),
                              on=["angle", "sim_idx"])
        corr["version"] = version
        frames.append(corr[["version", "angle", "sim_idx"] + list(SEPARATION)])
    return pd.concat(frames, ignore_index=True)


def separation_tests(per_system, angles):
    rows = []
    wide = per_system.pivot_table(index=["angle", "sim_idx"], columns="version", values=list(SEPARATION))
    for measure in SEPARATION:
        for angle in angles:
            w = wide.loc[angle, measure]
            ap, tb = w.all_pairs, w.top_bottom
            a_mean, a_lo, a_hi = mean_ci(ap)
            b_mean, b_lo, b_hi = mean_ci(tb)
            rows.append({"measure": measure, "angle": angle, "all_pairs_mean": a_mean, "all_pairs_ci_low": a_lo,
                         "all_pairs_ci_high": a_hi, "top_bottom_mean": b_mean, "top_bottom_ci_low": b_lo,
                         "top_bottom_ci_high": b_hi, "systems_top_bottom_higher": int((tb > ap).sum()),
                         **replicate_test(ap, tb, paired=True)})
    return pd.DataFrame(rows)


def plot_separation(per_system, tests, angles, path):
    measures = list(SEPARATION)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    for ax, measure in zip(axes.flat, measures):
        title, null = SEPARATION[measure]
        for i, angle in enumerate(angles):
            sub = per_system[per_system.angle == angle].pivot(index="sim_idx", columns="version", values=measure)
            x0, x1 = 2 * i, 2 * i + 0.8
            for _, r in sub.iterrows():
                ax.plot([x0, x1], [r.all_pairs, r.top_bottom], color="#BBBBBB", linewidth=0.7, zorder=1)
            ax.scatter(np.full(len(sub), x0), sub.all_pairs, s=12, color="#7F7F7F", zorder=2)
            ax.scatter(np.full(len(sub), x1), sub.top_bottom, s=12, color="#1B9E77", zorder=2)
            t = tests[(tests.measure == measure) & (tests.angle == angle)].iloc[0]
            ax.text(x0 + 0.4, 1.0, f"{t.mean_b - t.mean_a:+.3f}\n{p_text(t.nonparametric_p)}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
        ax.axhline(null, color="k", linestyle="--", linewidth=0.8)
        ax.set_xticks([2 * i + d for i in range(len(angles)) for d in (0, 0.8)],
                      [f"{a}\n{v}" for a in angles for v in ("all pairs", "top-bottom")], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.margins(y=0.25)
    fig.suptitle("Force-cluster separation by all-pairs vs top-bottom betweenness (lines join the same system; "
                 "text = mean change and Wilcoxon p)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261116, help="Jitter seed.")
    args = parser.parse_args()
    if not IS_TOP_BOTTOM:
        raise ValueError("Run with BC_DATASET=periodic_force_top_bottom")

    particles = particle_properties(load_particles())
    angles = sorted(sorted_angles(particles.angle), key=angle_key)
    tables, figures = OUT / "tables", OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    systems = system_statistics(particles)
    systems.to_csv(tables / "simulation_group_statistics.csv", index=False)
    summary_table(systems).to_csv(tables / "simulation_group_summary.csv", index=False)
    tests = comparisons(systems, angles)
    tests.to_csv(tables / "statistical_comparisons.csv", index=False)
    plot_simulation_means(systems, tests, angles, figures / "simulation_mean_boxplots", args.seed)

    per_system = separation_per_system()
    per_system.to_csv(tables / "separation_top_bottom_vs_all_pairs_per_system.csv", index=False)
    sep_tests = separation_tests(per_system, angles)
    sep_tests.to_csv(tables / "separation_top_bottom_vs_all_pairs.csv", index=False)
    plot_separation(per_system, sep_tests, angles, figures / "separation_top_bottom_vs_all_pairs.png")

    rho = particles.groupby(["angle", "sim_idx"])[["top_bottom", "all_pairs"]].corr(method="spearman")
    rho = rho.xs("top_bottom", level=2).all_pairs.rename("spearman_top_bottom_vs_all_pairs").reset_index()
    rho.to_csv(tables / "spearman_top_bottom_vs_all_pairs_per_system.csv", index=False)

    (OUT / "metadata.json").write_text(json.dumps({
        "dataset": DATASET_NAME, "input": str(PARTICLES_CSV),
        "all_pairs_results": str(ALL_PAIRS_ROOT), "properties": PROPERTIES, "scopes": SCOPES,
        "statistics": list(STATISTICS),
        "replicates": "systems (20 per angle); one value per system and group",
        "within_angle_test": "paired t and Wilcoxon (high vs non of the same system); effect = mean / SD of difference",
        "between_angle_test": "Welch t, Mann-Whitney U, Hedges g on system values (30deg minus 0deg)",
        "separation": "folder 5 (all pairs) vs this folder (top-bottom), steps 02 and 05 per-system measures, "
                      "paired over systems",
    }, indent=2))

    show = tests[(tests.statistic == "mean") & tests.property.isin(["top_bottom", "top_bottom_rel", "all_pairs_rel"])]
    print(show[["scope", "property", "comparison", "mean_a", "mean_b", "effect_size", "nonparametric_p"]]
          .round(5).to_string(index=False))
    print(sep_tests[["measure", "angle", "all_pairs_mean", "top_bottom_mean", "systems_top_bottom_higher",
                     "nonparametric_p"]].round(4).to_string(index=False))
    print(rho.groupby("angle").spearman_top_bottom_vs_all_pairs.agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
