#!/usr/bin/env python3
"""Betweenness rank composition and enrichment within radial bands (spatial control).

Betweenness in these packings falls steeply with distance from the axis, and force
clusters sit at different radii at different angles. To compare particles only with
others at a similar radius, every system is split into radial bands and betweenness
is ranked within each band (percentile 100 = highest in that band).

Radial position is r / R: distance from the axis (the system's mean x, y) divided by
the container radius R, taken as the largest r in the system (about 0.685 mm, the same
at every angle). Height is not used.

Band modes:
  count:  equal-count bands of relative radius (same number of particles per band);
  radius: fixed fractions of the local radius (e.g. r < R/2 and r >= R/2).
Enrichment = high-force fraction / the high-force share of that system's band, so a
band that simply holds more force-cluster particles does not count as enrichment.

Continuous version (profile_normalized/): the expected betweenness at each relative
radius, b(r), is the mean over all particles of an angle in equal-count radius bins,
linearly interpolated; each particle's betweenness / b(r) is then ranked within its
system and analyzed as in step 02.
Systems are the replicates: mean and 95% t interval over systems.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_only_common import (GROUP_COLORS, GROUP_LABELS, OUTPUT_ROOT, PARTICLES_CSV, angle_colors,
                              load_particles, sorted_angles, summarize)

OUT = OUTPUT_ROOT / "07_radial_control"
CONFIGS = (("count", 2), ("count", 4), ("radius", 2), ("radius", 4))
PROFILE_BINS = 12


def relative_radius(particles):
    frames = []
    for _, sub in particles.groupby(["angle", "sim_idx"]):
        sub = sub.copy()
        r = np.hypot(sub.x - sub.x.mean(), sub.y - sub.y.mean())
        sub["r"] = r
        sub["rel_r"] = r / r.max()
        sub["global_percentile"] = sub.betweenness.rank(pct=True)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def assign_bands(particles, mode, n_bands):
    frame = particles.copy()
    if mode == "count":
        frame["band"] = frame.groupby(["angle", "sim_idx"]).rel_r.transform(
            lambda v: pd.qcut(v.rank(method="first"), n_bands, labels=False))
    else:
        frame["band"] = np.minimum((frame.rel_r * n_bands).astype(int), n_bands - 1)
    keys = ["angle", "sim_idx", "band"]
    # Percentile within band, 0 = lowest betweenness, 1 = highest; ties share a rank.
    rank = frame.groupby(keys).betweenness.rank(method="average")
    frame["band_percentile"] = (rank - 0.5) / frame.groupby(keys).betweenness.transform("size")
    frame["is_high"] = (frame.group == "high_force").astype(float)
    frame["band_share"] = frame.groupby(keys).is_high.transform("mean")
    frame["system_share"] = frame.groupby(["angle", "sim_idx"]).is_high.transform("mean")
    return frame


def band_labels(mode, n_bands):
    edges = np.linspace(0, 1, n_bands + 1)
    if mode == "radius":
        return [f"r/R {edges[i]:.2g}-{edges[i + 1]:.2g}" for i in range(n_bands)]
    names = {2: ["inner half", "outer half"], 4: ["inner quarter", "2nd quarter", "3rd quarter", "outer quarter"]}
    return [f"{names.get(n_bands, [f'band {i + 1}' for i in range(n_bands)])[i]} (equal count)"
            for i in range(n_bands)]


def per_system(frame, bin_percent):
    n_bins = int(round(100 / bin_percent))
    frame = frame.assign(pbin=np.minimum((frame.band_percentile * n_bins).astype(int), n_bins - 1))
    profile, bins, cumulative, pooled = [], [], [], []
    for (angle, sim, band), sub in frame.groupby(["angle", "sim_idx", "band"]):
        share = sub.band_share.iat[0]
        profile.append({"angle": angle, "sim_idx": sim, "band": band, "particles": len(sub),
                        "band_share": share, "enrichment_band": share / sub.system_share.iat[0],
                        "mean_global_percentile": sub.global_percentile.mean(),
                        "mean_rel_r": sub.rel_r.mean()})
        if share == 0:
            continue
        for pbin, part in sub.groupby("pbin"):
            bins.append({"angle": angle, "sim_idx": sim, "band": band, "pbin": pbin, "particles": len(part),
                         "fraction_high_force": part.is_high.mean(),
                         "enrichment_high_force": part.is_high.mean() / share})
        for p in np.arange(0, 100, bin_percent):
            part = sub[sub.band_percentile >= p / 100]
            cumulative.append({"angle": angle, "sim_idx": sim, "band": band, "percentile": float(p),
                               "particles": len(part), "fraction_high_force": part.is_high.mean(),
                               "enrichment_high_force": part.is_high.mean() / share})
    # All bands together, each particle ranked only within its own band.
    for (angle, sim), sub in frame.groupby(["angle", "sim_idx"]):
        share = sub.system_share.iat[0]
        for pbin, part in sub.groupby("pbin"):
            pooled.append({"angle": angle, "sim_idx": sim, "pbin": pbin, "particles": len(part),
                           "fraction_high_force": part.is_high.mean(),
                           "enrichment_high_force": part.is_high.mean() / share,
                           "kind": "per_bin"})
        for p in np.arange(0, 100, bin_percent):
            part = sub[sub.band_percentile >= p / 100]
            pooled.append({"angle": angle, "sim_idx": sim, "percentile": float(p), "particles": len(part),
                           "fraction_high_force": part.is_high.mean(),
                           "enrichment_high_force": part.is_high.mean() / share, "kind": "cumulative"})
    return (pd.DataFrame(profile), pd.DataFrame(bins), pd.DataFrame(cumulative), pd.DataFrame(pooled))


def ci_line(ax, summary, x, column, color, label):
    ax.plot(x, summary[f"{column}_mean"], marker="o", markersize=3, color=color, label=label)
    ax.fill_between(x, summary[f"{column}_ci_low"], summary[f"{column}_ci_high"], color=color, alpha=0.18)


def plot_profile(profile, angles, labels, path):
    colors = angle_colors(angles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), squeeze=False)
    for angle in angles:
        summary = summarize(profile[profile.angle == angle], ["band"],
                            ["enrichment_band", "mean_global_percentile"])
        x = summary.band.to_numpy()
        ci_line(axes[0, 0], summary, x, "enrichment_band", colors[angle], angle)
        ci_line(axes[0, 1], summary, x, "mean_global_percentile", colors[angle], angle)
    axes[0, 0].axhline(1, color="k", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("High-force share of band / system share")
    axes[0, 0].set_title("Where force-cluster particles sit")
    axes[0, 1].axhline(0.5, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Mean betweenness percentile (whole system)")
    axes[0, 1].set_title("Where betweenness is high")
    for ax in axes[0]:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_xlabel("Radial band (inner = near axis, outer = near side wall)")
    axes[0, 0].legend(title="Angle", fontsize=8)
    fig.suptitle("Radial profiles (mean with 95% CI over systems; core only)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def bin_ticks(n_bins, bin_percent):
    ticks = list(range(0, n_bins, max(1, n_bins // 5)))
    return ticks, [f"{t * bin_percent:g}-{(t + 1) * bin_percent:g}" for t in ticks]


def plot_composition_by_band(bins, profile, angles, labels, bin_percent, path):
    n_bins = int(round(100 / bin_percent))
    n_bands = len(labels)
    fig, axes = plt.subplots(n_bands, len(angles), figsize=(4.6 * len(angles), 3.4 * n_bands),
                             sharey=True, squeeze=False)
    for row in range(n_bands):
        for col, angle in enumerate(angles):
            ax = axes[row, col]
            sub = bins[(bins.angle == angle) & (bins.band == row)]
            high = sub.groupby("pbin").fraction_high_force.mean().reindex(range(n_bins)).to_numpy()
            ax.bar(range(n_bins), high, width=0.9, color=GROUP_COLORS["high_force"],
                   label=GROUP_LABELS["high_force"])
            ax.bar(range(n_bins), 1 - high, bottom=high, width=0.9, color=GROUP_COLORS["non_high_force"],
                   label=GROUP_LABELS["non_high_force"])
            share = profile[(profile.angle == angle) & (profile.band == row)].band_share.mean()
            ax.axhline(share, color="k", linestyle="--", linewidth=1, label="Mean band high-force share")
            ticks, ticklabels = bin_ticks(n_bins, bin_percent)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, rotation=30)
            ax.set_ylim(0, 1)
            ax.set_title(f"{angle}, {labels[row]}", fontsize=10)
            if row == n_bands - 1:
                ax.set_xlabel("Betweenness percentile bin within band")
        axes[row, 0].set_ylabel("Fraction of particles in bin")
    axes[0, -1].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle("Composition of betweenness percentile bins within radial bands "
                 "(right = highest betweenness in band; mean over systems; core only)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_enrichment_by_band(bins, cumulative, angles, labels, bin_percent, path):
    colors = angle_colors(angles)
    n_bands = len(labels)
    fig, axes = plt.subplots(2, n_bands, figsize=(5.2 * n_bands, 8.4), sharey="row", squeeze=False)
    for col in range(n_bands):
        for angle in angles:
            sub = bins[(bins.angle == angle) & (bins.band == col)]
            summary = summarize(sub, ["pbin"], ["enrichment_high_force"])
            ci_line(axes[0, col], summary, (summary.pbin + 0.5) * bin_percent, "enrichment_high_force",
                    colors[angle], angle)
            sub = cumulative[(cumulative.angle == angle) & (cumulative.band == col)]
            summary = summarize(sub, ["percentile"], ["enrichment_high_force"])
            ci_line(axes[1, col], summary, summary.percentile, "enrichment_high_force", colors[angle], angle)
        axes[0, col].set_title(f"{labels[col]}: per percentile bin", fontsize=10)
        axes[1, col].set_title(f"{labels[col]}: at or above percentile p", fontsize=10)
        axes[0, col].set_xlabel("Betweenness percentile bin center (within band)")
        axes[1, col].set_xlabel("Betweenness percentile p within band")
    for ax in axes.ravel():
        ax.axhline(1, color="k", linestyle="--", linewidth=1)
    for row in range(2):
        axes[row, 0].set_ylabel("Enrichment (fraction / band share)")
    axes[0, 0].legend(title="Angle", fontsize=8)
    fig.suptitle("High-force over-representation by betweenness, ranked within radial bands "
                 "(1 = band share; band = 95% CI over systems)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_pooled(pooled, angles, bin_percent, path, ranked_by="ranked within radial band",
                title="Radius-controlled over-representation (all bands pooled; band = 95% CI over systems)"):
    """Same layout as 02's rank_enrichment_percentile, with ranks taken within radial bands."""
    colors = angle_colors(angles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), squeeze=False)
    for angle in angles:
        sub = pooled[(pooled.angle == angle) & (pooled.kind == "per_bin")]
        summary = summarize(sub, ["pbin"], ["enrichment_high_force"])
        ci_line(axes[0, 0], summary, (summary.pbin + 0.5) * bin_percent, "enrichment_high_force",
                colors[angle], angle)
        sub = pooled[(pooled.angle == angle) & (pooled.kind == "cumulative")]
        summary = summarize(sub, ["percentile"], ["enrichment_high_force"])
        ci_line(axes[0, 1], summary, summary.percentile, "enrichment_high_force", colors[angle], angle)
    for ax in axes[0]:
        ax.axhline(1, color="k", linestyle="--", linewidth=1)
        ax.set_ylabel("Enrichment (fraction / system share)")
    axes[0, 0].set_title("High-force particles: per percentile bin")
    axes[0, 0].set_xlabel(f"Betweenness percentile bin center ({ranked_by})")
    axes[0, 1].set_title("High-force particles: at or above percentile p")
    axes[0, 1].set_xlabel(f"Betweenness percentile p ({ranked_by})")
    axes[0, 0].legend(title="Angle", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def radius_profile(particles, reference="pooled"):
    """Expected betweenness b(r): mean betweenness in equal-count rel_r bins, interpolated to every
    particle. reference='pooled': bins and b(r) from all systems of an angle together;
    reference='own': from each system's own particles."""
    frames, edges = [], {}
    keys = "angle" if reference == "pooled" else ["angle", "sim_idx"]
    for angle, sub in particles.groupby(keys):
        sub = sub.copy()
        e = np.quantile(sub.rel_r, np.linspace(0, 1, PROFILE_BINS + 1))
        e[-1] += 1e-12
        sub["rbin"] = np.clip(np.searchsorted(e, sub.rel_r, side="right") - 1, 0, PROFILE_BINS - 1)
        centers = sub.groupby("rbin").rel_r.mean().to_numpy()
        means = sub.groupby("rbin").betweenness.mean().to_numpy()
        sub["expected_betweenness"] = np.interp(sub.rel_r, centers, means)
        sub["normalized_betweenness"] = sub.betweenness / sub.expected_betweenness
        frames.append(sub)
        edges[angle] = e
    frame = pd.concat(frames, ignore_index=True)
    frame["band"] = 0
    frame["band_percentile"] = ((frame.groupby(["angle", "sim_idx"]).normalized_betweenness.rank(method="average")
                                 - 0.5) / frame.groupby(["angle", "sim_idx"]).normalized_betweenness
                                .transform("size"))
    frame["is_high"] = (frame.group == "high_force").astype(float)
    frame["system_share"] = frame.groupby(["angle", "sim_idx"]).is_high.transform("mean")
    frame["band_share"] = frame.system_share
    return frame, edges


def plot_radius_profile(frame, angles, path, reference_note="expected = all systems of the angle pooled"):
    """Mean betweenness against relative radius, all particles and each group (per-system bin means)."""
    groups = (("all", None, "k", "All particles"),
              ("high_force", "high_force", GROUP_COLORS["high_force"], GROUP_LABELS["high_force"]),
              ("non_high_force", "non_high_force", GROUP_COLORS["non_high_force"],
               GROUP_LABELS["non_high_force"]))
    fig, axes = plt.subplots(2, len(angles), figsize=(4.4 * len(angles), 8), sharex=True, squeeze=False)
    rows = []
    for col, angle in enumerate(angles):
        sub = frame[frame.angle == angle]
        for name, group, color, label in groups:
            part = sub if group is None else sub[sub.group == group]
            per_sys = part.groupby(["sim_idx", "rbin"]).agg(rel_r=("rel_r", "mean"),
                                                             betweenness=("betweenness", "mean"),
                                                             normalized=("normalized_betweenness", "mean"))
            per_sys = per_sys.reset_index()
            summary = summarize(per_sys, ["rbin"], ["rel_r", "betweenness", "normalized"])
            rows.append(summary.assign(angle=angle, particles=name))
            ci_line(axes[0, col], summary, summary.rel_r_mean, "betweenness", color, label)
            if group is not None:
                ci_line(axes[1, col], summary, summary.rel_r_mean, "normalized", color, label)
        axes[0, col].set_title(angle)
        axes[1, col].axhline(1, color="k", linestyle="--", linewidth=1)
        axes[1, col].set_xlabel("Relative radius r / R (R = container radius)")
    axes[0, 0].set_ylabel("Mean betweenness")
    axes[1, 0].set_ylabel("Mean betweenness / expected at that radius")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Betweenness against radius ({PROFILE_BINS} equal-count radius bins; "
                 f"mean with 95% CI over systems; core only)\nBottom: betweenness / expected at that radius "
                 f"({reference_note})")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin-percent", type=float, default=10.0,
                        help="Percentile bin width within a band; bands hold fewer particles than a "
                             "system, so the default is coarser than step 02's 5%%.")
    args = parser.parse_args()
    if abs(100 / args.bin_percent - round(100 / args.bin_percent)) > 1e-9:
        raise ValueError("--bin-percent must divide 100")

    particles = relative_radius(load_particles())
    angles = sorted_angles(particles.angle)
    summary_rows = []
    for mode, n_bands in CONFIGS:
        frame = assign_bands(particles, mode, n_bands)
        labels = band_labels(mode, n_bands)
        profile, bins, cumulative, pooled = per_system(frame, args.bin_percent)
        out = OUT / f"{mode}_{n_bands}bands"
        tables, figures = out / "tables", out / "figures"
        tables.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        profile.to_csv(tables / "band_profile_per_system.csv", index=False)
        bins.to_csv(tables / "percentile_bins_per_system.csv", index=False)
        cumulative.to_csv(tables / "cumulative_per_system.csv", index=False)
        pooled.to_csv(tables / "pooled_radius_controlled_per_system.csv", index=False)
        summarize(profile, ["angle", "band"], ["particles", "band_share", "enrichment_band",
                                               "mean_global_percentile"]).to_csv(
            tables / "band_profile_summary.csv", index=False)

        plot_profile(profile, angles, labels, figures / "radial_profile.png")
        plot_composition_by_band(bins, profile, angles, labels, args.bin_percent,
                                 figures / "composition_by_band_percentile.png")
        plot_enrichment_by_band(bins, cumulative, angles, labels, args.bin_percent,
                                figures / "enrichment_by_band_percentile.png")
        plot_pooled(pooled, angles, args.bin_percent, figures / "enrichment_radius_controlled_percentile.png")

        top = 100 - 2 * args.bin_percent  # top 20% at the default bin width
        for angle in angles:
            for band in range(n_bands):
                values = cumulative[(cumulative.angle == angle) & (cumulative.band == band)
                                    & np.isclose(cumulative.percentile, top)]
                row = summarize(values, ["angle"], ["enrichment_high_force"]).iloc[0]
                summary_rows.append({"config": out.name, "band": labels[band], "angle": angle,
                                     "top_percent": 100 - top, "enrichment": row.enrichment_high_force_mean,
                                     "ci_low": row.enrichment_high_force_ci_low,
                                     "ci_high": row.enrichment_high_force_ci_high})
            values = pooled[(pooled.angle == angle) & (pooled.kind == "cumulative")
                            & np.isclose(pooled.percentile, top)]
            row = summarize(values, ["angle"], ["enrichment_high_force"]).iloc[0]
            summary_rows.append({"config": out.name, "band": "all bands (ranked within band)", "angle": angle,
                                 "top_percent": 100 - top, "enrichment": row.enrichment_high_force_mean,
                                 "ci_low": row.enrichment_high_force_ci_low,
                                 "ci_high": row.enrichment_high_force_ci_high})

    references = (("pooled", "", "expected = all systems of the angle pooled"),
                  ("own", "_own_reference", "expected = each system's own radius profile"))
    out = OUT / "profile_normalized"
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for reference, suffix, note in references:
        frame, _ = radius_profile(particles, reference)
        _, _, _, pooled = per_system(frame, args.bin_percent)
        frame[["angle", "sim_idx", "node_id", "group", "rel_r", "betweenness", "expected_betweenness",
               "normalized_betweenness", "band_percentile"]].rename(
            columns={"band_percentile": "normalized_percentile"}).to_csv(
            tables / f"particles_normalized{suffix}.csv", index=False)
        pooled.to_csv(tables / f"normalized_rank_per_system{suffix}.csv", index=False)
        plot_radius_profile(frame, angles, figures / f"betweenness_vs_radius{suffix}.png", note).to_csv(
            tables / f"betweenness_vs_radius_summary{suffix}.csv", index=False)
        plot_pooled(pooled, angles, args.bin_percent,
                    figures / f"enrichment_profile_normalized_percentile{suffix}.png",
                    ranked_by="of betweenness / expected at radius",
                    title=f"Radius-normalized over-representation ({note}; band = 95% CI over systems)")
        top = 100 - 2 * args.bin_percent
        for angle in angles:
            values = pooled[(pooled.angle == angle) & (pooled.kind == "cumulative")
                            & np.isclose(pooled.percentile, top)]
            row = summarize(values, ["angle"], ["enrichment_high_force"]).iloc[0]
            summary_rows.append({"config": out.name + suffix, "band": "betweenness / expected at radius",
                                 "angle": angle, "top_percent": 100 - top,
                                 "enrichment": row.enrichment_high_force_mean,
                                 "ci_low": row.enrichment_high_force_ci_low,
                                 "ci_high": row.enrichment_high_force_ci_high})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "top_enrichment_summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "input": str(PARTICLES_CSV), "bin_percent": args.bin_percent,
        "relative_radius": "r from the system's mean x, y, divided by the system's largest r "
                           "(container radius); height not used",
        "configs": [f"{m}_{n}bands" for m, n in CONFIGS] + ["profile_normalized",
                                                              "profile_normalized_own_reference"],
        "profile_normalized_own_reference": "as profile_normalized, but bins and expected betweenness from "
                                            "each system's own particles (files with suffix _own_reference)",
        "profile_normalized": f"betweenness / mean betweenness of the angle at the same relative radius "
                              f"({PROFILE_BINS} equal-count bins, linear interpolation), ranked within system",
        "percentile": "betweenness rank within (system, band); 100 = highest",
        "enrichment": "high-force fraction / high-force share of that system's band "
                      "(pooled figure: / system share)",
        "replicates": "systems; mean and 95% t interval over systems per angle",
    }, indent=2))
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
