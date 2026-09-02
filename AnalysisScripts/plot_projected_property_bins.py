#!/usr/bin/env python
"""
Quick projected bin plots for RubySim graph node properties.

The graph_feature_arrays table contains graph-level arrays only.  For spatial
x-z / y-z property maps this script uses the node_features table generated in
the same GraphPipeline directory.
"""

from __future__ import print_function

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/yfjin/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_GRAPH_FEATURE_ARRAYS = Path(
    "/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/"
    "AnalysisResults/PeriodicBoudaries/2026-08-03/GraphPipeline/graph_feature_arrays.pkl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/abucsek_root/abucsek0/yfjin/Granular_RubySim/"
    "AnalysisResults/PeriodicBoudaries/2026-08-03/Tests/visualization"
)
DEFAULT_BIN_SIZE = 1.0e-4
DEFAULT_COLOR_LIMITS = {
    "stress_hydro": (-1.6e-9, 0.0),
    "degree": (4.25, 5.75),
    "betweenness": (0.0025, 0.0065),
}
PROPERTY_EXCLUDE_COLUMNS = {
    "geometry",
    "sim_idx",
    "node_id",
    "x",
    "y",
    "z",
    "is_wall",
}


def _parse_csv_list(value):
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_color_limits(value):
    limits = dict(DEFAULT_COLOR_LIMITS)
    if value is None:
        return limits
    for spec in _parse_csv_list(value):
        try:
            prop, raw_range = spec.split(":", 1)
            raw_vmin, raw_vmax = raw_range.split(":", 1)
            vmin = float(raw_vmin)
            vmax = float(raw_vmax)
        except ValueError:
            raise ValueError(
                "Invalid --color-limits entry {0!r}; use property:vmin:vmax".format(spec)
            )
        if vmax <= vmin:
            raise ValueError(
                "Invalid --color-limits range for {0}: vmax must be greater than vmin".format(
                    prop
                )
            )
        limits[prop] = (vmin, vmax)
    return limits


def _resolve_node_file(node_file, graph_feature_arrays):
    if node_file is not None:
        return Path(node_file)

    graph_feature_arrays = Path(graph_feature_arrays)
    sibling = graph_feature_arrays.with_name("node_features.csv")
    if sibling.exists():
        return sibling

    raise FileNotFoundError(
        "Could not find node_features.csv next to {0}. Provide --node-file.".format(
            graph_feature_arrays
        )
    )


def _bin_statistic_2d(x, z, values, x_edges, z_edges, stat):
    valid = np.isfinite(x) & np.isfinite(z)
    if stat != "count":
        valid &= np.isfinite(values)

    x = x[valid]
    z = z[valid]
    values = values[valid]

    counts, _, _ = np.histogram2d(z, x, bins=[z_edges, x_edges])
    if stat == "count":
        out = counts.astype(float)
        out[out == 0] = np.nan
        return out

    sums, _, _ = np.histogram2d(z, x, bins=[z_edges, x_edges], weights=values)
    if stat == "sum":
        sums[counts == 0] = np.nan
        return sums

    if stat == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / counts
        means[counts == 0] = np.nan
        return means

    if stat == "median":
        xi = np.searchsorted(x_edges, x, side="right") - 1
        zi = np.searchsorted(z_edges, z, side="right") - 1
        inside = (xi >= 0) & (xi < len(x_edges) - 1) & (zi >= 0) & (zi < len(z_edges) - 1)
        out = np.full((len(z_edges) - 1, len(x_edges) - 1), np.nan, dtype=float)
        for iz in range(len(z_edges) - 1):
            for ix in range(len(x_edges) - 1):
                vals = values[inside & (xi == ix) & (zi == iz)]
                if vals.size:
                    out[iz, ix] = np.median(vals)
        return out

    raise ValueError("Unsupported statistic: {0}".format(stat))


def _finite_limits(series):
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(vals.min()), float(vals.max())


def _edges_from_bin_size(limits, bin_size):
    lo, hi = limits
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite bin limits: {0}".format(limits))
    if hi <= lo:
        pad = max(abs(lo) * 0.01, 1.0e-12)
        lo -= pad
        hi += pad
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    start = np.floor(lo / bin_size) * bin_size
    stop = np.ceil(hi / bin_size) * bin_size
    n_bins = max(1, int(np.ceil((stop - start) / bin_size)))
    return start + np.arange(n_bins + 1, dtype=float) * bin_size


def _bin_size_from_target_bins(df, projection, target_bins):
    coord = "x" if projection == "xz" else "y"
    coord_limits = _finite_limits(df[coord])
    z_limits = _finite_limits(df["z"])
    if coord_limits is None or z_limits is None:
        raise ValueError("Missing finite coordinates for projection {0}".format(projection))

    coord_span = coord_limits[1] - coord_limits[0]
    z_span = z_limits[1] - z_limits[0]
    return max(coord_span, z_span) / float(target_bins)


def _format_geometry_list(values):
    def key(label):
        text = str(label)
        if text.endswith("deg"):
            try:
                return (0, float(text[:-3]))
            except ValueError:
                pass
        return (1, text)

    return sorted([str(v) for v in values], key=key)


def _select_properties(df, requested):
    numeric_cols = [
        col
        for col in df.columns
        if col not in PROPERTY_EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    if requested is None or [item.lower() for item in requested] == ["all"]:
        return numeric_cols, []

    missing = [prop for prop in requested if prop not in df.columns]
    selected = [
        prop
        for prop in requested
        if prop in df.columns and pd.api.types.is_numeric_dtype(df[prop])
    ]
    non_numeric = [
        prop
        for prop in requested
        if prop in df.columns and not pd.api.types.is_numeric_dtype(df[prop])
    ]
    return selected, missing + non_numeric


def _geometry_envelope(sub, coord, coord_edges):
    x = pd.to_numeric(sub[coord], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(sub["z"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(z)
    x = x[valid]
    z = z[valid]
    if x.size == 0:
        return None, None, None

    centers = 0.5 * (coord_edges[:-1] + coord_edges[1:])
    lower = np.full(len(centers), np.nan, dtype=float)
    upper = np.full(len(centers), np.nan, dtype=float)
    idx = np.searchsorted(coord_edges, x, side="right") - 1
    inside = (idx >= 0) & (idx < len(centers))
    for i in range(len(centers)):
        vals = z[inside & (idx == i)]
        if vals.size:
            lower[i] = np.nanmin(vals)
            upper[i] = np.nanmax(vals)
    return centers, lower, upper


def _apply_mean_std_clim(images):
    arrays = [img.get_array().compressed() for img in images if img.get_array().compressed().size]
    if not arrays:
        return None
    finite_values = np.concatenate(arrays)
    mean = float(np.nanmean(finite_values))
    std = float(np.nanstd(finite_values))
    vmin = mean - 2.0 * std
    vmax = mean + 2.0 * std
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        for img in images:
            img.set_clim(vmin, vmax)
        return mean, std, vmin, vmax
    return mean, std, None, None


def _apply_manual_clim(images, vmin, vmax):
    for img in images:
        img.set_clim(vmin, vmax)
    return vmin, vmax


def _plot_position_diagnostic(df, geometries, out_dir):
    fig, axes = plt.subplots(
        len(geometries),
        2,
        figsize=(10.0, max(3.4 * len(geometries), 4.0)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, geometry in enumerate(geometries):
        sub = df[df["geometry"].astype(str) == str(geometry)]
        for col, coord in enumerate(["x", "y"]):
            ax = axes[row, col]
            ax.hexbin(sub[coord], sub["z"], gridsize=80, cmap="magma", mincnt=1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("{0} {1}-z positions".format(geometry, coord))
            ax.set_xlabel(coord)
            ax.set_ylabel("z")
    out_path = out_dir / "position_projection_diagnostic.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_property(
    df,
    prop,
    projection,
    geometries,
    bin_size,
    stat,
    out_dir,
    cmap,
    overlay_geometry,
    color_limits,
):
    coord = "x" if projection == "xz" else "y"
    x_label = coord
    z_label = "z"

    fig, axes = plt.subplots(
        1,
        len(geometries),
        figsize=(5.2 * len(geometries), 4.6),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]

    coord_limits = _finite_limits(df[coord])
    z_limits = _finite_limits(df["z"])
    if coord_limits is None or z_limits is None:
        raise ValueError("Missing finite coordinates for projection {0}".format(projection))

    coord_edges = _edges_from_bin_size(coord_limits, bin_size)
    z_edges = _edges_from_bin_size(z_limits, bin_size)

    images = []
    for ax, geometry in zip(axes, geometries):
        sub = df[df["geometry"].astype(str) == str(geometry)]
        binned = _bin_statistic_2d(
            pd.to_numeric(sub[coord], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(sub["z"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(sub[prop], errors="coerce").to_numpy(dtype=float),
            coord_edges,
            z_edges,
            stat,
        )
        img = ax.imshow(
            binned,
            origin="lower",
            extent=[coord_edges[0], coord_edges[-1], z_edges[0], z_edges[-1]],
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
        )
        images.append(img)
        ax.set_title(str(geometry))
        ax.set_xlabel(x_label)
        ax.set_ylabel(z_label)
        ax.set_aspect("equal", adjustable="box")
        if overlay_geometry:
            envelope = _geometry_envelope(sub, coord, coord_edges)
            centers, lower, upper = envelope
            if centers is not None:
                ax.plot(centers, lower, color="white", linewidth=1.0, alpha=0.95)
                ax.plot(centers, upper, color="white", linewidth=1.0, alpha=0.95)

    manual_limits = color_limits.get(prop)
    if manual_limits is not None:
        _apply_manual_clim(images, manual_limits[0], manual_limits[1])
        clim_label = "manual [{0:g}, {1:g}]".format(manual_limits[0], manual_limits[1])
    else:
        clim = _apply_mean_std_clim(images)
        clim_label = "mean +/- 2 std" if clim is not None and clim[2] is not None else None

    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.92)
    if clim_label is not None:
        cbar.set_label("{0} ({1}); limits = {2}".format(prop, stat, clim_label))
    else:
        cbar.set_label("{0} ({1})".format(prop, stat))
    fig.suptitle("{0} projection: {1}".format(projection.upper(), prop))

    out_path = out_dir / "{0}_{1}_{2}_bins.png".format(prop, projection, stat)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path, len(coord_edges) - 1, len(z_edges) - 1, bin_size


def main():
    parser = argparse.ArgumentParser(
        description="Make x-z/y-z binned heatmaps of graph node properties."
    )
    parser.add_argument(
        "--graph-feature-arrays",
        default=str(DEFAULT_GRAPH_FEATURE_ARRAYS),
        help="Graph feature arrays path; used to locate sibling node_features.csv.",
    )
    parser.add_argument("--node-file", default=None, help="Optional node_features.csv path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--properties",
        default="all",
        help="Comma-separated node properties to plot, or 'all' for all numeric properties.",
    )
    parser.add_argument(
        "--projection",
        choices=["xz", "yz", "both"],
        default="both",
        help="Projection to plot.",
    )
    parser.add_argument(
        "--geometries",
        default=None,
        help="Comma-separated geometry labels. Defaults to all found, e.g. 0deg,30deg.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=None,
        help="Physical bin size used for both projected coordinate and z.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Optional target bins along the longest axis. Used only when --bin-size is omitted.",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "median", "sum", "count"],
        default="mean",
        help="Statistic shown in each spatial bin.",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument(
        "--color-limits",
        default=None,
        help=(
            "Comma-separated manual color limits as property:vmin:vmax. "
            "Defaults include stress_hydro:-0.1e-9:0, degree:4:6, betweenness:0.0025:0.0065."
        ),
    )
    parser.add_argument(
        "--no-geometry-overlay",
        action="store_true",
        help="Disable position-derived geometry envelope overlay.",
    )
    parser.add_argument(
        "--no-position-diagnostic",
        action="store_true",
        help="Do not save the x-z/y-z position diagnostic figure.",
    )
    args = parser.parse_args()

    node_file = _resolve_node_file(args.node_file, args.graph_feature_arrays)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_properties = _parse_csv_list(args.properties)
    projections = ["xz", "yz"] if args.projection == "both" else [args.projection]

    df = pd.read_csv(node_file)
    color_limits = _parse_color_limits(args.color_limits)
    properties, skipped_properties = _select_properties(df, requested_properties)
    if not properties:
        raise ValueError("None of the requested properties were found in {0}".format(node_file))

    if "is_wall" in df.columns:
        df = df[df["is_wall"] != True]
    geometries = _parse_csv_list(args.geometries)
    if geometries is None:
        geometries = _format_geometry_list(df["geometry"].dropna().unique())
    else:
        df = df[df["geometry"].astype(str).isin([str(g) for g in geometries])]

    diagnostic_path = None
    if not args.no_position_diagnostic:
        diagnostic_path = _plot_position_diagnostic(df, geometries, out_dir)

    saved = []
    for prop in properties:
        for projection in projections:
            if args.bin_size is not None:
                bin_size = args.bin_size
            elif args.bins is not None:
                bin_size = _bin_size_from_target_bins(df, projection, args.bins)
            else:
                bin_size = DEFAULT_BIN_SIZE
            saved.append(
                _plot_property(
                    df=df,
                    prop=prop,
                    projection=projection,
                    geometries=geometries,
                    bin_size=bin_size,
                    stat=args.stat,
                    out_dir=out_dir,
                    cmap=args.cmap,
                    overlay_geometry=not args.no_geometry_overlay,
                    color_limits=color_limits,
                )
            )

    print("Read node data from: {0}".format(node_file))
    print("Geometries: {0}".format(", ".join(geometries)))
    print("Properties: {0}".format(", ".join(properties)))
    if skipped_properties:
        print("Skipped missing/non-numeric properties: {0}".format(", ".join(skipped_properties)))
    used_manual_limits = {prop: color_limits[prop] for prop in properties if prop in color_limits}
    if used_manual_limits:
        print("Manual color limits:")
        for prop, (vmin, vmax) in sorted(used_manual_limits.items()):
            print("  {0}: {1:g} to {2:g}".format(prop, vmin, vmax))
    if diagnostic_path is not None:
        print("Saved position diagnostic: {0}".format(diagnostic_path))
    print("Saved {0} figure(s) in: {1}".format(len(saved), out_dir))
    for path, n_coord, n_z, bin_size in saved:
        print("{0} ({1} x {2} bins, bin_size={3:g})".format(path, n_coord, n_z, bin_size))


if __name__ == "__main__":
    main()
