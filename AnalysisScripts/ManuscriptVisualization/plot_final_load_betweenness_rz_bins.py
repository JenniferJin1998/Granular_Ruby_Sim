#!/usr/bin/env python3
"""Export manuscript r-z bin plots of final-load node betweenness.

Binning follows the r-z bin-plot workflow in PostAnalysis/PresentationFigurePlots.ipynb
(section 2.4.8): particles from all systems of one orientation are pooled, centered on
their mean position, and binned in (r, z_eff), where inclined cells follow the
specimen boundary angle. Non-zero angles use the dual-slope (V-shaped) cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yfjin/manuscript-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

PROJECT = Path(__file__).resolve().parents[2]
CASE = PROJECT / "AnalysisResults" / "final_load"
DEFAULT_INPUT = CASE / "0_graph_and_basic_stats" / "graph_data" / "graph_dict_labeled.pkl"
DEFAULT_OUTPUT = CASE / "Visualization" / "betweenness_rz_bins"
PROPERTY = "betweenness"
N_R_BINS = 13
N_Z_BINS = 10
COLOR_LIMITS = (0.005, 0.045)
# Dual-slope split radius (m) used for inclined geometries in the notebook figures.
SPLIT_R = 3.75e-4
TEXT_FONTSIZE = 10.5
M_TO_MM = 1000.0
# Layout in inches. Same scale for every figure: 1 mm of r or z = INCHES_PER_MM.
INCHES_PER_MM = 3.5
FRAME_PAD_MM = 0.04
LAYOUT_WITH_AXES = (0.65, 0.55, 0.12, 0.42)  # left, bottom, top, gap between panels
LAYOUT_NO_AXES = (0.05, 0.05, 0.05, 0.08)
CBAR_GAP = 0.15
CBAR_WIDTH = 0.14
CBAR_LABEL_SPACE = 0.85
CBAR_HEIGHT_FRACTION = 0.7
VARIANTS = {"with_axes": True, "no_axes": False}


def angle_from_label(label):
    match = re.search(r"(\d+(?:\.\d+)?)", str(label))
    if match is None:
        raise ValueError(f"Cannot parse angle from geometry label: {label}")
    return float(match.group(1))


def load_particles(graph_dict, geometry, graph_view):
    positions, values = [], []
    for graph in graph_dict[geometry][graph_view]:
        for _, data in graph.nodes(data=True):
            if data.get("is_wall", False):
                continue
            value = float(data[PROPERTY])
            position = np.asarray(data["position"], dtype=float)
            if np.isfinite(value) and position.shape == (3,) and np.isfinite(position).all():
                positions.append(position)
                values.append(value)
    if not positions:
        raise ValueError(f"No finite {PROPERTY} values for {geometry}")
    return np.asarray(positions), np.asarray(values)


def z_eff_from_rz(r, z, k, split_r):
    left = z - k * r
    right = z + k * r - 2.0 * k * split_r
    return np.where(r <= split_r, left, right)


def bin_rz(positions, values, angle_deg, split_r):
    """Mean property per (z_eff, r) cell; returns the physical corner mesh too."""
    centered = positions - positions.mean(axis=0)
    r = np.hypot(centered[:, 0], centered[:, 1])
    z = centered[:, 2] - centered[:, 2].min()
    k = math.tan(math.radians(angle_deg))
    z_eff = z_eff_from_rz(r, z, k, split_r)

    r_edges = np.linspace(r.min(), r.max(), N_R_BINS + 1)
    z_eff_edges = np.linspace(z_eff.min(), z_eff.max(), N_Z_BINS + 1)
    r_idx = np.clip(np.digitize(r, r_edges) - 1, 0, N_R_BINS - 1)
    z_idx = np.clip(np.digitize(z_eff, z_eff_edges) - 1, 0, N_Z_BINS - 1)

    sums = np.zeros((N_Z_BINS, N_R_BINS))
    counts = np.zeros((N_Z_BINS, N_R_BINS), dtype=int)
    np.add.at(sums, (z_idx, r_idx), values)
    np.add.at(counts, (z_idx, r_idx), 1)
    mean = np.full_like(sums, np.nan)
    mean[counts > 0] = sums[counts > 0] / counts[counts > 0]

    r_mesh, z_eff_mesh = np.meshgrid(r_edges, z_eff_edges)
    left = z_eff_mesh + k * r_mesh
    right = z_eff_mesh - k * r_mesh + 2.0 * k * split_r
    z_mesh = np.where(r_mesh <= split_r, left, right)
    return {"mean": mean, "counts": counts, "r_mesh": r_mesh, "z_mesh": z_mesh, "k": k}


def draw_bins(ax, binned, cmap, vmin, vmax, r_limits, z_span, show_axes):
    """Equal r/z scaling; the specimen is centered vertically in a shared-size frame."""
    r_mesh = binned["r_mesh"] * M_TO_MM
    z_mesh = binned["z_mesh"] * M_TO_MM
    mesh = ax.pcolormesh(r_mesh, z_mesh, binned["mean"], shading="flat",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    z_top, z_bottom = z_mesh.min(), z_mesh.max()
    z_mid = 0.5 * (z_top + z_bottom)
    ax.set_xlim(*r_limits)
    ax.set_ylim(z_mid + 0.5 * z_span, z_mid - 0.5 * z_span)
    ax.set_aspect("equal")
    if not show_axes:
        ax.axis("off")
        return mesh
    # Ticks only where the specimen exists, so padding never shows negative z.
    ax.set_xticks(np.arange(0.0, r_mesh.max() + 1e-9, 0.2))
    ax.set_yticks(np.arange(0.0, z_bottom + 1e-9, 0.2))
    ax.set_xlabel("r (mm)", fontsize=TEXT_FONTSIZE)
    ax.tick_params(labelsize=TEXT_FONTSIZE - 1)
    return mesh


def make_figure(panels, cmap, vmin, vmax, r_limits, z_span, show_axes):
    """Fixed physical layout: every output shares the same inches per mm and centering."""
    left, bottom, top, gap = LAYOUT_WITH_AXES if show_axes else LAYOUT_NO_AXES
    ax_w = (r_limits[1] - r_limits[0]) * INCHES_PER_MM
    ax_h = z_span * INCHES_PER_MM
    cbar_x = left + len(panels) * ax_w + (len(panels) - 1) * gap + CBAR_GAP
    fig_w = cbar_x + CBAR_WIDTH + CBAR_LABEL_SPACE
    fig_h = bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, binned in enumerate(panels):
        ax = fig.add_axes([(left + i * (ax_w + gap)) / fig_w, bottom / fig_h,
                           ax_w / fig_w, ax_h / fig_h])
        mesh = draw_bins(ax, binned, cmap, vmin, vmax, r_limits, z_span, show_axes)
        if show_axes and i == 0:
            ax.set_ylabel("z (mm)", fontsize=TEXT_FONTSIZE)
    cbar_h = CBAR_HEIGHT_FRACTION * ax_h
    cax = fig.add_axes([cbar_x / fig_w, (bottom + 0.5 * (ax_h - cbar_h)) / fig_h,
                        CBAR_WIDTH / fig_w, cbar_h / fig_h])
    cbar = fig.colorbar(mesh, cax=cax, extend="both")
    cbar.set_label("Betweenness", fontsize=TEXT_FONTSIZE)
    cbar.ax.tick_params(labelsize=TEXT_FONTSIZE - 1)
    return fig


def save_figure(fig, directory, stem, formats, dpi):
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angles", nargs="+", default=None, help="e.g. 0deg 30deg; default: all")
    parser.add_argument("--graph-view", default="core", choices=["core", "full"])
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--vmin", type=float, default=COLOR_LIMITS[0])
    parser.add_argument("--vmax", type=float, default=COLOR_LIMITS[1])
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        graph_dict = pickle.load(handle)
    geometries = sorted(args.angles or graph_dict, key=angle_from_label)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input": str(args.input), "input_sha256": sha256(args.input),
        "property": PROPERTY, "graph_view": args.graph_view, "statistic": "mean",
        "n_r_bins": N_R_BINS, "n_z_bins": N_Z_BINS, "split_r_m": SPLIT_R,
        "cmap": args.cmap, "color_limits": [args.vmin, args.vmax],
        "units": "mm; r from pooled particle centroid, z from lowest particle, axis inverted",
        "geometries": {},
    }
    results, n_systems = {}, {}
    for geometry in geometries:
        positions, values = load_particles(graph_dict, geometry, args.graph_view)
        results[geometry] = (values, bin_rz(positions, values, angle_from_label(geometry), SPLIT_R))
        n_systems[geometry] = len(graph_dict[geometry][args.graph_view])
    # The graphs take several GB; release them before rendering high-dpi figures.
    del graph_dict
    # One frame size for every angle, so all panels share the same physical scale.
    r_max = max(b["r_mesh"].max() for _, b in results.values()) * M_TO_MM
    z_height = max(np.ptp(b["z_mesh"]) for _, b in results.values()) * M_TO_MM
    r_limits = (-FRAME_PAD_MM, r_max + FRAME_PAD_MM)
    z_span = z_height + 2.0 * FRAME_PAD_MM
    manifest["layout"] = {
        "r_limits_mm": list(r_limits), "z_frame_height_mm": z_span,
        "vertical_alignment": "each specimen centered in the frame",
        "aspect": "equal", "inches_per_mm": INCHES_PER_MM, "dpi": args.dpi,
    }
    plot_args = (args.cmap, args.vmin, args.vmax, r_limits, z_span)

    for geometry, (values, binned) in results.items():
        stem = f"final_load__{PROPERTY}__rz_bins__{geometry}"
        outputs = []
        for variant, show_axes in VARIANTS.items():
            fig = make_figure([binned], *plot_args, show_axes)
            outputs += save_figure(fig, args.output_root / variant,
                                   f"{stem}__{variant}", args.formats, args.dpi)

        finite = binned["mean"][np.isfinite(binned["mean"])]
        manifest["geometries"][geometry] = {
            "systems": n_systems[geometry], "particles": int(values.size),
            "incline_k": binned["k"], "empty_bins": int((binned["counts"] == 0).sum()),
            "min_particles_per_bin": int(binned["counts"][binned["counts"] > 0].min()),
            "bin_mean_range": [float(finite.min()), float(finite.max())],
            "bins_clipped_low": int((finite < args.vmin).sum()),
            "bins_clipped_high": int((finite > args.vmax).sum()),
            "outputs": outputs,
        }
        print(f"{geometry}: {values.size} particles, bin means "
              f"{finite.min():.4f}-{finite.max():.4f} -> {stem}__*")

    row_stem = f"final_load__{PROPERTY}__rz_bins__{'_'.join(geometries)}"
    manifest["row_outputs"] = []
    for variant, show_axes in VARIANTS.items():
        fig = make_figure([b for _, b in results.values()], *plot_args, show_axes)
        manifest["row_outputs"] += save_figure(fig, args.output_root / "all_angles_row",
                                               f"{row_stem}__{variant}", args.formats, args.dpi)
    print(f"Row figures -> {args.output_root / 'all_angles_row'}")

    manifest_path = args.output_root / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
