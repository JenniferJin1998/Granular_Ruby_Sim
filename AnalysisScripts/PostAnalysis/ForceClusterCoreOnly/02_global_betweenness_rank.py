#!/usr/bin/env python3
"""Share of high-force particles across betweenness ranks within each system.

Within every system (464 particles), betweenness is ranked and cut into equal
percentile bins (bin 0 = highest betweenness). For each bin, and for each
cumulative top k%, we report the high-force fraction and its enrichment: that
fraction divided by the system's own high-force share (1 = no preference). The
share differs strongly between systems, so enrichment is what compares fairly.
Systems are the replicates: mean and 95% t interval over the systems of an angle.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_only_common import (GROUP_COLORS, GROUP_LABELS, GROUPS, OUTPUT_ROOT, PARTICLES_CSV,
                              angle_colors, load_particles, sorted_angles, summarize)

OUT = OUTPUT_ROOT / "02_global_betweenness_rank"


def add_ranks(particles, bin_percent):
    n_bins = int(round(100 / bin_percent))
    ranked = []
    for _, sub in particles.groupby(["angle", "sim_idx"]):
        sub = sub.copy()
        # Fraction of the system ranked at or above the particle (0 = top); ties share a rank.
        sub["top_fraction"] = (sub.betweenness.rank(ascending=False, method="average") - 0.5) / len(sub)
        sub["rank_bin"] = np.minimum((sub.top_fraction * n_bins).astype(int), n_bins - 1)
        sub["system_high_share"] = (sub.group == "high_force").mean()
        ranked.append(sub)
    return pd.concat(ranked, ignore_index=True)


def per_system(ranked, bin_percent):
    bins, topk = [], []
    for (angle, sim), sub in ranked.groupby(["angle", "sim_idx"]):
        share = sub.system_high_share.iat[0]
        for rank_bin, part in sub.groupby("rank_bin"):
            fraction = (part.group == "high_force").mean()
            bins.append({"angle": angle, "sim_idx": sim, "rank_bin": rank_bin, "particles": len(part),
                         "system_high_share": share, "fraction_high_force": fraction,
                         "enrichment_high_force": fraction / share})
        for k in np.arange(bin_percent, 100 + 1e-9, bin_percent):
            part = sub[sub.top_fraction < k / 100]
            fraction = (part.group == "high_force").mean()
            topk.append({"angle": angle, "sim_idx": sim, "top_percent": float(k), "particles": len(part),
                         "fraction_high_force": fraction, "enrichment_high_force": fraction / share})
    return pd.DataFrame(bins), pd.DataFrame(topk)


def bin_label(i, bin_percent):
    return f"{i * bin_percent:g}-{(i + 1) * bin_percent:g}"


def plot_composition(bins, angles, bin_percent, path):
    n_bins = int(bins.rank_bin.max()) + 1
    fig, axes = plt.subplots(1, len(angles), figsize=(4.6 * len(angles), 4.4), sharey=True, squeeze=False)
    for ax, angle in zip(axes[0], angles):
        sub = bins[bins.angle == angle]
        high = sub.groupby("rank_bin").fraction_high_force.mean().reindex(range(n_bins)).to_numpy()
        ax.bar(range(n_bins), high, width=0.9, color=GROUP_COLORS["high_force"],
               label=GROUP_LABELS["high_force"])
        ax.bar(range(n_bins), 1 - high, bottom=high, width=0.9, color=GROUP_COLORS["non_high_force"],
               label=GROUP_LABELS["non_high_force"])
        ax.axhline(sub.groupby("sim_idx").system_high_share.first().mean(), color="k", linestyle="--",
                   linewidth=1, label="Mean system high-force share")
        ticks = list(range(0, n_bins, max(1, n_bins // 5)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([bin_label(t, bin_percent) for t in ticks], rotation=30)
        ax.set_title(angle)
        ax.set_xlabel("Betweenness rank bin within system (top %)")
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("Fraction of particles in bin (mean over systems)")
    axes[0, -1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Composition of betweenness rank bins (left = highest betweenness; core only)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_enrichment(bins, topk, angles, bin_percent, path):
    colors = angle_colors(angles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), squeeze=False)
    panels = ((bins, "rank_bin", "Betweenness rank bin center (top %)", "per rank bin"),
              (topk, "top_percent", "Top k% of betweenness within system", "cumulative top k%"))
    for ax, (frame, x_column, x_label, title) in zip(axes[0], panels):
        for angle in angles:
            summary = summarize(frame[frame.angle == angle], [x_column], ["enrichment_high_force"])
            x = summary[x_column].to_numpy(dtype=float)
            if x_column == "rank_bin":
                x = (x + 0.5) * bin_percent
            ax.plot(x, summary.enrichment_high_force_mean, marker="o", markersize=3,
                    color=colors[angle], label=angle)
            ax.fill_between(x, summary.enrichment_high_force_ci_low, summary.enrichment_high_force_ci_high,
                            color=colors[angle], alpha=0.18)
        ax.axhline(1, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"High-force particles: {title}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Enrichment (fraction / system share)")
    axes[0, 0].legend(title="Angle", fontsize=8)
    fig.suptitle("Over-representation in betweenness ranks (1 = system share; band = 95% CI over systems)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin-percent", type=float, default=5.0,
                        help="Width of rank bins and step of cumulative top-k%%; must divide 100.")
    args = parser.parse_args()
    if abs(100 / args.bin_percent - round(100 / args.bin_percent)) > 1e-9:
        raise ValueError("--bin-percent must divide 100")

    particles = load_particles()
    angles = sorted_angles(particles.angle)
    ranked = add_ranks(particles, args.bin_percent)
    bins, topk = per_system(ranked, args.bin_percent)

    tables, figures = OUT / "tables", OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(tables / "particles_global_rank.csv", index=False)
    bins.to_csv(tables / "rank_bins_per_system.csv", index=False)
    topk.to_csv(tables / "top_k_per_system.csv", index=False)
    columns = ["fraction_high_force", "enrichment_high_force"]
    summarize(bins, ["angle", "rank_bin"], columns).to_csv(tables / "rank_bins_summary.csv", index=False)
    top_summary = summarize(topk, ["angle", "top_percent"], columns)
    top_summary.to_csv(tables / "top_k_summary.csv", index=False)

    plot_composition(bins, angles, args.bin_percent, figures / "rank_bin_composition.png")
    plot_enrichment(bins, topk, angles, args.bin_percent, figures / "rank_enrichment.png")
    (OUT / "metadata.json").write_text(json.dumps({
        "input": str(PARTICLES_CSV), "bin_percent": args.bin_percent, "rank_bin_0": "highest betweenness",
        "ties": "average rank", "enrichment": "high-force fraction / that system's high-force share",
        "replicates": "systems; mean and 95% t interval over systems per angle",
    }, indent=2))

    print("Top-k enrichment of high-force particles, mean [95% CI]:")
    for k in (args.bin_percent, 2 * args.bin_percent, 4 * args.bin_percent):
        print(f"  top {k:g}%:")
        for _, r in top_summary[np.isclose(top_summary.top_percent, k)].iterrows():
            print(f"    {r.angle}: {r.enrichment_high_force_mean:.2f} "
                  f"[{r.enrichment_high_force_ci_low:.2f}, {r.enrichment_high_force_ci_high:.2f}]")


if __name__ == "__main__":
    main()
