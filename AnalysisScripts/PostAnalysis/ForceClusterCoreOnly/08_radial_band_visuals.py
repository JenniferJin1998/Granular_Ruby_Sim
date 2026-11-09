#!/usr/bin/env python3
"""Visual checks of the radial bands of step 07: how much radial betweenness gradient remains inside them.

1. Betweenness vs r / R with the band boundaries of both band definitions
   (equal-count quarters and fixed r/R rings), per-band box plots, and the
   within-band Spearman rho(r/R, betweenness) per system (mean and 95% CI over
   systems). A band that removes position well has rho near 0.
2. 3D views of one example system per angle: one panel per band. Particles in the
   band are colored by betweenness and their contacts drawn in gray; everything
   outside the band is light gray. The color range is fixed for all panels at the
   chosen percentiles of betweenness over all particles of all angles.

Radius definition and bands are those of 07_radial_control.py (core graph only).
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.stats import spearmanr

from core_only_common import OUTPUT_ROOT, load_particles, sorted_angles, summarize

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("radial_control", HERE / "07_radial_control.py")
radial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(radial)

OUT = OUTPUT_ROOT / "08_radial_band_visuals"
CONTACTS = OUTPUT_ROOT / "05_correlation" / "tables" / "contacts_edge_betweenness.csv"
N_BANDS = 4
MODES = ("count", "radius")
MODE_TITLES = {"count": "Equal-count quarters", "radius": "Fixed r/R rings"}
BAND_COLORS = ("#1b9e77", "#7570b3", "#e7298a", "#d95f02")
LIGHT = "#d9d9d9"
IN_EDGE = "#6e6e6e"


def band_frames(particles):
    return {mode: radial.assign_bands(particles, mode, N_BANDS) for mode in MODES}


def within_band_rho(frames):
    rows = []
    for mode, frame in frames.items():
        for (angle, sim, band), sub in frame.groupby(["angle", "sim_idx", "band"]):
            rho = spearmanr(sub.rel_r, sub.betweenness).statistic if len(sub) > 3 else np.nan
            rows.append({"mode": mode, "angle": angle, "sim_idx": sim, "band": band,
                         "particles": len(sub), "rel_r_min": sub.rel_r.min(), "rel_r_max": sub.rel_r.max(),
                         "spearman_rho": rho})
        for (angle, sim), sub in frame.groupby(["angle", "sim_idx"]):
            rows.append({"mode": "whole system", "angle": angle, "sim_idx": sim, "band": -1,
                         "particles": len(sub), "rel_r_min": 0.0, "rel_r_max": 1.0,
                         "spearman_rho": spearmanr(sub.rel_r, sub.betweenness).statistic})
    return pd.DataFrame(rows).drop_duplicates(["mode", "angle", "sim_idx", "band"])


def band_edges(frame, mode):
    """Mean band boundaries over systems (count mode varies slightly per system)."""
    if mode == "radius":
        return np.linspace(0, 1, N_BANDS + 1)[1:-1]
    cuts = frame.groupby(["angle", "sim_idx", "band"]).rel_r.max().groupby("band").mean()
    return cuts.to_numpy()[:-1]


def plot_scatter(frames, rho_summary, angles, labels, path, annotate=True, legend_outside=False):
    """annotate: median line and within-band rho; legend_outside: legends right of the last column."""
    fig, axes = plt.subplots(len(MODES), len(angles), figsize=(4.4 * len(angles), 4.0 * len(MODES)),
                             sharey=True, squeeze=False)
    y_top = max(f.betweenness.quantile(0.995) for f in frames.values())
    for row, mode in enumerate(MODES):
        frame = frames[mode]
        for ax, angle in zip(axes[row], angles):
            sub = frame[frame.angle == angle]
            for band in range(N_BANDS):
                part = sub[sub.band == band]
                ax.scatter(part.rel_r, part.betweenness, s=2, alpha=0.25, color=BAND_COLORS[band],
                           rasterized=True, label=labels[mode][band])
            if annotate:
                bins = pd.cut(sub.rel_r, np.linspace(0, 1, 21))
                median = sub.groupby(bins, observed=True).betweenness.median()
                ax.plot([b.mid for b in median.index], median.to_numpy(), color="k", linewidth=1.5,
                        label="Median (r/R bins of 0.05)")
            for edge in band_edges(sub, mode):
                ax.axvline(edge, color="k", linestyle="--", linewidth=0.8)
            for band in range(N_BANDS if annotate else 0):
                s = rho_summary[(rho_summary["mode"] == mode) & (rho_summary.angle == angle)
                                & (rho_summary.band == band)].iloc[0]
                x = sub[sub.band == band].rel_r.median()
                ax.text(x, y_top * 0.97, f"ρ={s.spearman_rho_mean:+.2f}", ha="center", va="top", fontsize=7,
                        color=BAND_COLORS[band], fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, y_top)
            ax.set_title(f"{angle}, {MODE_TITLES[mode]}", fontsize=10)
            if row == len(MODES) - 1:
                ax.set_xlabel("Relative radius r / R")
        axes[row, 0].set_ylabel("Betweenness")
        if legend_outside:
            axes[row, -1].legend(fontsize=8, markerscale=5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                 title=MODE_TITLES[mode], title_fontsize=8, frameon=False)
        else:
            axes[row, -1].legend(fontsize=7, markerscale=4, loc="center right")
    title = "Betweenness vs radius with band boundaries (all 20 systems per angle; dashed = band edge"
    title += ("; ρ = within-band Spearman ρ(r/R, betweenness), mean over systems)" if annotate else ")")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_boxes(frames, angles, labels, path):
    fig, axes = plt.subplots(len(MODES), len(angles), figsize=(4.4 * len(angles), 4.0 * len(MODES)),
                             sharey=True, squeeze=False)
    for row, mode in enumerate(MODES):
        frame = frames[mode]
        for ax, angle in zip(axes[row], angles):
            sub = frame[frame.angle == angle]
            data = [sub[sub.band == b].betweenness.to_numpy() for b in range(N_BANDS)]
            box = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
            for patch, color in zip(box["boxes"], BAND_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            counts = [len(d) / sub.sim_idx.nunique() for d in data]
            ax.set_xticks(range(1, N_BANDS + 1))
            ax.set_xticklabels([f"{labels[mode][b].replace(' (equal count)', '')}\n(~{counts[b]:.0f}/system)"
                                for b in range(N_BANDS)], fontsize=7, rotation=20)
            ax.set_title(f"{angle}, {MODE_TITLES[mode]}", fontsize=10)
        axes[row, 0].set_ylabel("Betweenness")
    fig.suptitle("Betweenness distribution in each radial band (box = quartiles, whiskers = 1.5 IQR; "
                 "all 20 systems per angle)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_rho(rho_summary, angles, labels, path):
    fig, axes = plt.subplots(1, len(angles), figsize=(4.4 * len(angles), 4.2), sharey=True, squeeze=False)
    width = 0.38
    for ax, angle in zip(axes[0], angles):
        whole = rho_summary[(rho_summary["mode"] == "whole system") & (rho_summary.angle == angle)].iloc[0]
        ax.axhline(whole.spearman_rho_mean, color="k", linestyle=":", linewidth=1.2,
                   label=f"Whole system ({whole.spearman_rho_mean:+.2f})")
        for i, mode in enumerate(MODES):
            s = rho_summary[(rho_summary["mode"] == mode) & (rho_summary.angle == angle)].sort_values("band")
            x = np.arange(N_BANDS) + (i - 0.5) * width
            err = np.vstack([s.spearman_rho_mean - s.spearman_rho_ci_low, s.spearman_rho_ci_high - s.spearman_rho_mean])
            ax.bar(x, s.spearman_rho_mean, width=width, yerr=err, capsize=2,
                   color=["#4d4d4d", "#bababa"][i], label=MODE_TITLES[mode])
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(range(N_BANDS))
        ax.set_xticklabels(["band 1\n(inner)", "band 2", "band 3", "band 4\n(outer)"], fontsize=8)
        ax.set_title(angle)
    axes[0, 0].set_ylabel("Spearman ρ(r/R, betweenness) within band")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Radial gradient left inside each band (0 = position removed; mean and 95% CI over 20 systems)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def pick_examples(particles, angles, requested):
    """One system per angle: the one whose high-force share is closest to the angle's median."""
    chosen = {}
    shares = particles.assign(h=particles.group == "high_force").groupby(["angle", "sim_idx"]).h.mean()
    for i, angle in enumerate(angles):
        if requested:
            chosen[angle] = requested[i]
        else:
            s = shares.loc[angle]
            chosen[angle] = int((s - s.median()).abs().idxmin())
    return chosen


def example_system(frame, contacts, angle, sim):
    nodes = frame[(frame.angle == angle) & (frame.sim_idx == sim)].copy()
    nodes["X"] = (nodes.x - nodes.x.mean()) * 1e3
    nodes["Y"] = (nodes.y - nodes.y.mean()) * 1e3
    nodes["Z"] = (nodes.z - nodes.z.min()) * 1e3
    edges = contacts[(contacts.angle == angle) & (contacts.sim_idx == sim)]
    return nodes, edges


def draw_band_3d(ax, nodes, edges, band, title, cmap, norm, title_color="k", title_y=None):
    pos = nodes.set_index("node_id")[["X", "Y", "Z"]]
    in_band = set(nodes.node_id[nodes.band == band])
    seg_in, seg_out = [], []
    for u, v in zip(edges.edge_u, edges.edge_v):
        seg = [pos.loc[u].to_numpy(), pos.loc[v].to_numpy()]
        (seg_in if (u in in_band and v in in_band) else seg_out).append(seg)
    ax.add_collection3d(Line3DCollection(seg_out, colors=LIGHT, linewidths=0.3, alpha=0.5))
    out = nodes[nodes.band != band]
    ax.scatter(out.X, out.Y, out.Z, s=4, color=LIGHT, alpha=0.35, depthshade=False)
    ax.add_collection3d(Line3DCollection(seg_in, colors=IN_EDGE, linewidths=0.7, alpha=0.9))
    inn = nodes[nodes.band == band]
    ax.scatter(inn.X, inn.Y, inn.Z, s=18, c=inn.betweenness, cmap=cmap, norm=norm,
               edgecolors="k", linewidths=0.2, depthshade=False)
    ax.set_box_aspect([np.ptp(nodes[c]) for c in ("X", "Y", "Z")])
    ax.view_init(elev=18, azim=-60)
    ax.set_xlabel("x (mm)", fontsize=6, labelpad=-8)
    ax.set_ylabel("y (mm)", fontsize=6, labelpad=-8)
    ax.set_zlabel("z (mm)", fontsize=6, labelpad=-8)
    ax.tick_params(labelsize=5, pad=-3)
    ax.set_title(title, fontsize=8, color=title_color, y=title_y)


def band_title(nodes, band, label):
    rr = nodes[nodes.band == band].rel_r
    return (f"{label.replace(' (equal count)', '')}\n"
            f"{int((nodes.band == band).sum())} particles, r/R {rr.min():.2f}-{rr.max():.2f}")


def add_colorbar(fig, cax, cmap, norm, clip):
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    cb.set_label(f"Betweenness (fixed range: {clip[0]:g}th-{clip[1]:g}th percentile of all particles)")


def plot_3d(frame, contacts, examples, angles, labels, mode, vmin, vmax, clip, path_all, per_angle_dir):
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(4.0 * N_BANDS + 1.5, 3.6 * len(angles)))
    fig.subplots_adjust(left=0.01, right=0.88, top=0.91, bottom=0.01, wspace=0.02, hspace=0.12)
    for r, angle in enumerate(angles):
        sim = examples[angle]
        nodes, edges = example_system(frame, contacts, angle, sim)
        for b in range(N_BANDS):
            ax = fig.add_subplot(len(angles), N_BANDS, r * N_BANDS + b + 1, projection="3d")
            draw_band_3d(ax, nodes, edges, b, f"{angle} (system {sim}), " + band_title(nodes, b, labels[b]),
                         cmap, norm)
    add_colorbar(fig, fig.add_axes([0.90, 0.25, 0.015, 0.5]), cmap, norm, clip)
    fig.suptitle(f"Betweenness by radial band, one example system per angle ({MODE_TITLES[mode]})\n"
                 "Colored = particles in band; dark gray = contacts within band; light gray = outside band",
                 fontsize=11)
    fig.savefig(path_all, dpi=250, bbox_inches="tight")
    fig.savefig(path_all.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    per_angle_dir.mkdir(parents=True, exist_ok=True)
    for angle in angles:
        sim = examples[angle]
        nodes, edges = example_system(frame, contacts, angle, sim)
        fig = plt.figure(figsize=(4.2 * N_BANDS + 1.5, 4.4))
        fig.subplots_adjust(left=0.01, right=0.88, top=0.80, bottom=0.02, wspace=0.02)
        for b in range(N_BANDS):
            ax = fig.add_subplot(1, N_BANDS, b + 1, projection="3d")
            draw_band_3d(ax, nodes, edges, b, band_title(nodes, b, labels[b]), cmap, norm)
        add_colorbar(fig, fig.add_axes([0.90, 0.15, 0.012, 0.6]), cmap, norm, clip)
        fig.suptitle(f"{angle}, system {sim}: betweenness by radial band ({MODE_TITLES[mode]})", fontsize=11)
        fig.savefig(per_angle_dir / f"{angle}_system{sim}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)


def plot_combined(frame, contacts, examples, angles, labels, mode, vmin, vmax, clip, y_top, path):
    """One column per angle: betweenness vs r/R (all systems) on top, then the example system's
    3D view for each band (inner to outer). Band colors of the scatter match the 3D panel titles."""
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(4.4 * len(angles) + 2.2, 3.3 + 3.3 * N_BANDS))
    top = fig.add_gridspec(1, len(angles), left=0.05, right=0.86, top=0.935, bottom=0.79, wspace=0.12)
    low = fig.add_gridspec(N_BANDS, len(angles), left=0.05, right=0.86, top=0.75, bottom=0.0,
                           wspace=0.12, hspace=0.0)
    first = None
    for c, angle in enumerate(angles):
        ax = fig.add_subplot(top[0, c], sharey=first)
        first = first or ax
        sub = frame[frame.angle == angle]
        for band in range(N_BANDS):
            part = sub[sub.band == band]
            ax.scatter(part.rel_r, part.betweenness, s=2, alpha=0.25, color=BAND_COLORS[band],
                       rasterized=True, label=labels[band].replace(" (equal count)", ""))
        for edge in band_edges(sub, mode):
            ax.axvline(edge, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, y_top)
        ax.set_xlabel("Relative radius r / R", fontsize=9)
        ax.set_title(f"{angle}: all 20 systems", fontsize=10)
        if c == 0:
            ax.set_ylabel("Betweenness")
        else:
            ax.tick_params(labelleft=False)
        if c == len(angles) - 1:
            ax.legend(fontsize=8, markerscale=5, loc="upper left", bbox_to_anchor=(1.03, 1.0),
                      title=MODE_TITLES[mode], title_fontsize=8, frameon=False)

        sim = examples[angle]
        nodes, edges = example_system(frame, contacts, angle, sim)
        for b in range(N_BANDS):
            ax3 = fig.add_subplot(low[b, c], projection="3d")
            draw_band_3d(ax3, nodes, edges, b, f"{angle} (system {sim}), " + band_title(nodes, b, labels[b]),
                         cmap, norm, title_color=BAND_COLORS[b], title_y=0.88)
    add_colorbar(fig, fig.add_axes([0.89, 0.10, 0.012, 0.55]), cmap, norm, clip)
    fig.suptitle(f"Betweenness by radial band ({MODE_TITLES[mode]}). Top: betweenness vs r/R, dashed = band edge. "
                 "Below: one example system per angle, one band per row (inner to outer);\n"
                 "colored = particles in band, dark gray = contacts within band, light gray = outside band",
                 fontsize=11)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", type=float, nargs=2, default=(5.0, 95.0),
                        help="Percentiles of betweenness (all particles, all angles) for the fixed color range.")
    parser.add_argument("--systems", type=int, nargs="*", default=None,
                        help="Example sim_idx per angle, in angle order (default: median high-force share).")
    parser.add_argument("--skip-3d", action="store_true", help="Only the distribution figures (fast).")
    args = parser.parse_args()

    particles = radial.relative_radius(load_particles())
    angles = sorted_angles(particles.angle)
    frames = band_frames(particles)
    labels = {mode: radial.band_labels(mode, N_BANDS) for mode in MODES}
    contacts = pd.read_csv(CONTACTS, usecols=["angle", "sim_idx", "edge_u", "edge_v"])

    tables, figures = OUT / "tables", OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rho = within_band_rho(frames)
    rho.to_csv(tables / "within_band_spearman_per_system.csv", index=False)
    rho_summary = summarize(rho, ["mode", "angle", "band"], ["spearman_rho", "particles", "rel_r_min", "rel_r_max"])
    rho_summary.to_csv(tables / "within_band_spearman_summary.csv", index=False)

    plot_scatter(frames, rho_summary, angles, labels, figures / "betweenness_vs_radius_bands.png")
    plot_scatter(frames, rho_summary, angles, labels, figures / "betweenness_vs_radius_bands_legend_outside.png",
                 legend_outside=True)
    plot_scatter(frames, rho_summary, angles, labels, figures / "betweenness_vs_radius_bands_plain.png",
                 annotate=False, legend_outside=True)
    plot_boxes(frames, angles, labels, figures / "betweenness_box_by_band.png")
    plot_rho(rho_summary, angles, labels, figures / "within_band_gradient.png")

    vmin, vmax = np.percentile(particles.betweenness, args.clip)
    examples = pick_examples(particles, angles, args.systems)
    for mode in MODES if not args.skip_3d else ():
        plot_3d(frames[mode], contacts, examples, angles, labels[mode], mode, vmin, vmax, args.clip,
                figures / f"band_3d_{mode}.png", figures / f"band_3d_{mode}_per_angle")
        y_top = max(f.betweenness.quantile(0.995) for f in frames.values())
        plot_combined(frames[mode], contacts, examples, angles, labels[mode], mode, vmin, vmax, args.clip,
                      y_top, figures / f"band_scatter_3d_{mode}.png")

    (OUT / "metadata.json").write_text(json.dumps({
        "bands": {m: f"{N_BANDS} bands, mode '{m}' as in 07_radial_control.py" for m in MODES},
        "radius": "r / R, R = largest particle-center r in the system",
        "color_range": {"percentiles": list(args.clip), "vmin": vmin, "vmax": vmax},
        "example_systems": examples, "contacts": str(CONTACTS),
    }, indent=2, default=str))

    print("Within-band Spearman rho(r/R, betweenness), mean over systems:")
    print(rho_summary.pivot_table(index=["angle"], columns=["mode", "band"], values="spearman_rho_mean").round(2)
          .to_string())
    print("Examples:", examples, f"color range {vmin:.4f}-{vmax:.4f}")


if __name__ == "__main__":
    main()
