#!/usr/bin/env python3
"""How well betweenness separates high-force from non-high-force particles at a given radius.

AUC = P(betweenness of a random high-force particle > that of a random non-high-force
particle), ties counted half (Mann-Whitney U / (n_high * n_non)); 0.5 = no separation.

1. AUC vs radius: within each relative-radius bin (equal-count bins of r / R, edges from all
   systems of an angle pooled, as in 07 profile_normalized), AUC per system, then mean and
   95% t interval over systems.
2. Whole-system summaries per system, compared across angles:
   - raw:                 betweenness over the whole system (as in step 05);
   - radius-stratified:   only pairs in the same radius bin are compared,
                          sum over bins of U / sum over bins of n_high * n_non;
   - normalized (pooled / own reference): betweenness / expected at that radius (step 07);
   - r/R alone:           the radial position as the score (> 0.5: high-force particles sit
                          farther from the axis), to show how much position alone separates.
Radius r / R as in 07_radial_control.py; core graph only; height not used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from core_only_common import OUTPUT_ROOT, angle_colors, load_particles, sorted_angles, summarize

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("radial_control", HERE / "07_radial_control.py")
radial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(radial)

OUT = OUTPUT_ROOT / "09_radial_auc"
MEASURES = (("auc_raw", "Raw betweenness (whole system)"),
            ("auc_stratified", "Radius-stratified (same-bin pairs only)"),
            ("auc_normalized_pooled", "Betweenness / expected at radius (pooled reference)"),
            ("auc_normalized_own", "Betweenness / expected at radius (own reference)"),
            ("auc_radius", "r/R alone (> 0.5: high-force farther out)"))
MEASURE_STYLE = {"auc_raw": ("#7f7f7f", "o", "-"), "auc_stratified": ("#000000", "s", "-"),
                 "auc_normalized_pooled": ("#1f77b4", "^", "-"), "auc_normalized_own": ("#17becf", "v", "--"),
                 "auc_radius": ("#d62728", "D", ":")}


def u_stat(score, is_high):
    """Mann-Whitney U of the high-force group (ties count half) and n_high * n_non."""
    n1, n0 = int(is_high.sum()), int((~is_high).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, 0
    return stats.mannwhitneyu(score[is_high], score[~is_high]).statistic, n1 * n0


def add_bins(particles, n_bins):
    frames = []
    for _, sub in particles.groupby("angle"):
        sub = sub.copy()
        edges = np.quantile(sub.rel_r, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-12
        sub["rbin"] = np.clip(np.searchsorted(edges, sub.rel_r, side="right") - 1, 0, n_bins - 1)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def measures(frame):
    by_bin, by_system = [], []
    for (angle, sim), sub in frame.groupby(["angle", "sim_idx"]):
        high = (sub.group == "high_force").to_numpy()
        u_sum = pairs_sum = 0.0
        for rbin, part in sub.groupby("rbin"):
            h = (part.group == "high_force").to_numpy()
            u, pairs = u_stat(part.betweenness.to_numpy(), h)
            if pairs:
                u_sum += u
                pairs_sum += pairs
            by_bin.append({"angle": angle, "sim_idx": sim, "rbin": rbin, "rel_r": part.rel_r.mean(),
                           "particles": len(part), "high_force": int(h.sum()),
                           "auc": u / pairs if pairs else np.nan})
        row = {"angle": angle, "sim_idx": sim, "auc_stratified": u_sum / pairs_sum}
        for column, score in (("auc_raw", "betweenness"), ("auc_normalized_pooled", "norm_pooled"),
                              ("auc_normalized_own", "norm_own"), ("auc_radius", "rel_r")):
            u, pairs = u_stat(sub[score].to_numpy(), high)
            row[column] = u / pairs
        by_system.append(row)
    return pd.DataFrame(by_bin), pd.DataFrame(by_system)


def plot_vs_radius(by_bin, angles, n_bins, path):
    colors = angle_colors(angles)
    fig, axes = plt.subplots(1, len(angles) + 1, figsize=(4.2 * (len(angles) + 1), 4.2), sharey=True,
                             squeeze=False)
    for ax, angle in zip(axes[0], angles):
        s = summarize(by_bin[by_bin.angle == angle], ["rbin"], ["auc", "rel_r"])
        ax.plot(s.rel_r_mean, s.auc_mean, marker="o", markersize=3, color=colors[angle])
        ax.fill_between(s.rel_r_mean, s.auc_ci_low, s.auc_ci_high, color=colors[angle], alpha=0.2)
        ax.set_title(angle)
    for angle in angles:
        s = summarize(by_bin[by_bin.angle == angle], ["rbin"], ["auc", "rel_r"])
        axes[0, -1].plot(s.rel_r_mean, s.auc_mean, marker="o", markersize=3, color=colors[angle], label=angle)
    axes[0, -1].set_title("All angles (means)")
    axes[0, -1].legend(fontsize=8)
    for ax in axes[0]:
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Relative radius r / R")
    axes[0, 0].set_ylabel("AUC of betweenness within radius bin")
    fig.suptitle(f"High-force vs non-high-force separation by betweenness at each radius "
                 f"({n_bins} equal-count radius bins; mean with 95% CI over systems; 0.5 = none)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_summary(by_system, angles, path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(angles))
    offsets = np.linspace(-0.24, 0.24, len(MEASURES))
    for (column, label), dx in zip(MEASURES, offsets):
        s = summarize(by_system, ["angle"], [column]).set_index("angle").loc[angles]
        color, marker, style = MEASURE_STYLE[column]
        err = np.vstack([s[f"{column}_mean"] - s[f"{column}_ci_low"], s[f"{column}_ci_high"] - s[f"{column}_mean"]])
        ax.errorbar(x + dx, s[f"{column}_mean"], yerr=err, color=color, marker=marker, linestyle=style,
                    capsize=3, label=label)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(angles)
    ax.set_xlabel("Boundary angle")
    ax.set_ylabel("AUC (mean and 95% CI over systems)")
    ax.set_title("Whole-system AUC for high-force vs non-high-force particles, with and without radius control",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bins", type=int, default=12, help="Equal-count radius bins per angle.")
    args = parser.parse_args()

    particles = radial.relative_radius(load_particles())
    angles = sorted_angles(particles.angle)
    keys = ["angle", "sim_idx", "node_id"]
    for reference in ("pooled", "own"):
        normalized, _ = radial.radius_profile(particles, reference)
        particles = particles.merge(normalized[keys + ["normalized_betweenness"]].rename(
            columns={"normalized_betweenness": f"norm_{reference}"}), on=keys, validate="one_to_one")
    frame = add_bins(particles, args.bins)
    by_bin, by_system = measures(frame)

    tables, figures = OUT / "tables", OUT / "figures"
    tag = "" if args.bins == 12 else f"_{args.bins}bins"  # default run keeps the plain file names
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    by_bin.to_csv(tables / f"auc_by_radius_bin_per_system{tag}.csv", index=False)
    summarize(by_bin, ["angle", "rbin"], ["auc", "rel_r", "particles", "high_force"]).to_csv(
        tables / f"auc_by_radius_bin_summary{tag}.csv", index=False)
    by_system.to_csv(tables / f"auc_per_system{tag}.csv", index=False)
    columns = [c for c, _ in MEASURES]
    summary = summarize(by_system, ["angle"], columns)
    summary.to_csv(tables / f"auc_summary{tag}.csv", index=False)

    plot_vs_radius(by_bin, angles, args.bins, figures / f"auc_vs_radius{tag}.png")
    plot_summary(by_system, angles, figures / f"auc_summary_by_angle{tag}.png")
    (OUT / f"metadata{tag}.json").write_text(json.dumps({
        "auc": "Mann-Whitney U / (n_high * n_non), ties half; per system, then mean and 95% t CI over systems",
        "radius_bins": f"{args.bins} equal-count bins of r/R per angle (all systems pooled)",
        "measures": dict(MEASURES),
    }, indent=2))

    with pd.option_context("display.width", 200):
        print(summary[["angle"] + [f"{c}_{k}" for c in columns for k in ("mean", "ci_low", "ci_high")]]
              .round(3).to_string(index=False))
        print(summarize(by_bin, ["angle", "rbin"], ["auc", "rel_r"])
              .pivot(index="rbin", columns="angle", values="auc_mean").round(3).to_string())


if __name__ == "__main__":
    main()
