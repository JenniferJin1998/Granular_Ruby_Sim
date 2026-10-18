#!/usr/bin/env python3
"""Export individual manuscript figures of final-load strong contact networks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yfjin/manuscript-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

PROJECT = Path(__file__).resolve().parents[2]
CASE = PROJECT / "AnalysisResults" / "final_load"
NORM = Normalize(10, 80, clip=True)
CMAP = matplotlib.colormaps["inferno"]
HIGH_FORCE_LINEWIDTH = 1.4
TEXT_FONTSIZE = 10.5
LEGEND_SYMBOL_SIZE = 4.5
VIEWS = {"along_x": (0, 0), "along_y": (0, 90), "perspective_3d": (22, -55),
         "perspective_3d_cylinder": (22, -55)}


def cylinder_envelope(graph, geometry):
    """Fit a plain cylinder radius to sidewall contact coordinates."""
    positions, low, high = geometry
    sidewall = []
    for u, v, data in graph.edges(data=True):
        walls = [n for n in (u, v) if graph.nodes[n].get("is_wall", False)]
        if walls and abs(float(data["n_unit"][2])) < 0.1:
            sidewall.append(positions[walls[0]][:2])
    xy = np.asarray(sidewall)
    if len(xy) < 3:
        raise ValueError("Cylinder fitting requires at least three sidewall contacts")
    design = np.column_stack((2 * xy, np.ones(len(xy))))
    solution, _, rank, _ = np.linalg.lstsq(design, np.sum(xy ** 2, axis=1), rcond=None)
    if rank < 3:
        raise ValueError("Sidewall contacts do not define a circle")
    center = solution[:2]
    fitted_radius = float(np.sqrt(solution[2] + np.dot(center, center)))
    radius = fitted_radius
    return {"center_xy_mm": center.tolist(), "radius_mm": radius,
            "fitted_sidewall_radius_mm": fitted_radius,
            "z_limits_mm": [float(low[2]), float(high[2])], "sidewall_contact_count": len(xy),
            "sidewall_rms_residual_mm": float(np.sqrt(np.mean((np.linalg.norm(xy - center, axis=1) - radius) ** 2)))}


def draw_cylinder(ax, envelope):
    cx, cy = envelope["center_xy_mm"]
    radius = envelope["radius_mm"]
    bottom, top = envelope["z_limits_mm"]
    theta = np.linspace(0, 2 * np.pi, 97)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    ax.plot_surface(np.tile(x, (2, 1)), np.tile(y, (2, 1)),
                    np.array([[bottom] * len(theta), [top] * len(theta)]),
                    color="#8c9daa", alpha=0.045, shade=False, linewidth=0,
                    antialiased=False, rstride=1, cstride=1, zorder=0)
    for z in (bottom, top):
        ax.plot(x, y, np.full_like(theta, z), color="#71808b", linewidth=0.7, alpha=0.65, zorder=5)
    # Silhouette generators for azimuth -55 degrees, matching the 3D camera.
    for angle in np.deg2rad([35, 215]):
        ax.plot([cx + radius * np.cos(angle)] * 2, [cy + radius * np.sin(angle)] * 2,
                [bottom, top], color="#71808b", linewidth=0.7, alpha=0.55, zorder=5)


def prepare_geometry(graph):
    """Resolve wall contact points without inventing a common wall centroid."""
    positions = {}
    for node, data in graph.nodes(data=True):
        if not data.get("is_wall", False):
            position = np.asarray(data["position"], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"Invalid particle position: {node}")
            positions[node] = position
    for u, v, data in graph.edges(data=True):
        for node in (u, v):
            if graph.nodes[node].get("is_wall", False):
                position = np.asarray(data["contact_location"], dtype=float)
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError(f"Invalid wall contact point: {node}")
                if node in positions and not np.allclose(positions[node], position, atol=1e-12, rtol=0):
                    raise ValueError("A wall node has multiple contact positions")
                positions[node] = position
        if "is_high_force" not in data:
            raise ValueError("Input must contain existing is_high_force labels")
        angle = float(data["angle_with_zz"])
        if not np.isfinite(angle) or not 0 <= angle <= 90:
            raise ValueError(f"Invalid angle_with_zz: {angle}")
    if set(positions) != set(graph):
        raise ValueError("Input contains wall nodes without contact locations")
    # Identical coordinates and limits for both wall variants and all views.
    origin = np.min(list(positions.values()), axis=0)
    positions = {node: (point - origin) * 1000 for node, point in positions.items()}
    points = np.asarray(list(positions.values()))
    low, high = points.min(axis=0), points.max(axis=0)
    return positions, low, high


def visible_network(graph, positions, include_walls):
    nodes = [node for node in graph if include_walls or not graph.nodes[node].get("is_wall", False)]
    visible = set(nodes)
    weak, strong, angles = [], [], []
    strongest = {}
    wall_edges = strong_wall_edges = 0
    for u, v, data in graph.edges(data=True):
        if u not in visible or v not in visible:
            continue
        segment = np.asarray([positions[u], positions[v]])
        wall = graph.nodes[u].get("is_wall", False) or graph.nodes[v].get("is_wall", False)
        wall_edges += int(wall)
        if bool(data["is_high_force"]):
            strong.append(segment)
            angle = float(data["angle_with_zz"])
            angles.append(angle)
            strong_wall_edges += int(wall)
            for node in (u, v):
                force = float(data["normal_force"])
                if node not in strongest or force > strongest[node][0]:
                    strongest[node] = (force, angle)
        else:
            weak.append(segment)
    return nodes, weak, strong, angles, strongest, wall_edges, strong_wall_edges


def draw_figure(graph, geometry, network, view, include_walls, target, formats, dpi):
    positions, low, high = geometry
    nodes, weak, strong, angles, strongest, _, _ = network
    fig = plt.figure(figsize=(4.7, 4.4), facecolor="white")
    ax = fig.add_axes((0.00, 0.16, 0.83, 0.83), projection="3d", computed_zorder=False)
    if view == "perspective_3d_cylinder":
        envelope = cylinder_envelope(graph, geometry)
        draw_cylinder(ax, envelope)
        cx, cy = envelope["center_xy_mm"]
        radius = envelope["radius_mm"]
        low = np.minimum(low, [cx - radius, cy - radius, low[2]])
        high = np.maximum(high, [cx + radius, cy + radius, high[2]])
    if weak:
        ax.add_collection3d(Line3DCollection(weak, colors="#bcbcbc", linewidths=0.45, alpha=0.22, zorder=1))
    if strong:
        ax.add_collection3d(Line3DCollection(strong, colors=CMAP(NORM(angles)), linewidths=HIGH_FORCE_LINEWIDTH, alpha=1, zorder=3))
    for wall, marker in ((False, "o"), (True, "^")):
        for highlighted in (False, True):
            group = [n for n in nodes if bool(graph.nodes[n].get("is_wall", False)) == wall
                     and (n in strongest) == highlighted]
            if not group:
                continue
            xyz = np.asarray([positions[n] for n in group])
            colors = [CMAP(NORM(strongest[n][1])) for n in group] if highlighted else "#bcbcbc"
            ax.scatter(*xyz.T, c=colors, marker=marker, s=(15 if wall else 8) if highlighted else (9 if wall else 3),
                       alpha=1 if highlighted else 0.38, edgecolors="none", depthshade=False,
                       zorder=4 if highlighted else 2)
    spans = high - low
    padding = max(spans) * 0.035
    ax.set(xlim=(low[0] - padding, high[0] + padding),
           ylim=(low[1] - padding, high[1] + padding),
           zlim=(low[2] - padding, high[2] + padding))
    ax.set_box_aspect(spans + 2 * padding, zoom=1.45)
    ax.set_proj_type("ortho")
    ax.view_init(*VIEWS[view])
    ax.set_axis_off()
    fig.canvas.draw()
    extent_points = np.asarray([positions[node] for node in nodes])
    if view == "perspective_3d_cylinder":
        theta = np.linspace(0, 2 * np.pi, 97)
        envelope_points = np.asarray([
            [cx + radius * np.cos(angle), cy + radius * np.sin(angle), height]
            for height in (low[2], high[2]) for angle in theta
        ])
        extent_points = np.vstack((extent_points, envelope_points))
    projected_x, projected_y, _ = proj3d.proj_transform(*extent_points.T, ax.get_proj())
    projected = ax.transData.transform(np.column_stack((projected_x, projected_y)))
    system_bbox = Bbox.from_extents(*projected.min(axis=0), *projected.max(axis=0))
    system_fraction = system_bbox.transformed(fig.transFigure.inverted())
    cax = fig.add_axes((system_fraction.x1 + 0.035, system_fraction.y0,
                        0.019, system_fraction.height))
    colorbar = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax,
                            ticks=np.arange(10, 81, 10), extend="both", extendfrac=0.035)
    colorbar.set_label("Contact angle w.r.t.\nloading direction (°)",
                       fontsize=TEXT_FONTSIZE, labelpad=4)
    colorbar.ax.tick_params(labelsize=TEXT_FONTSIZE, width=0.6, length=2)
    colorbar.outline.set_linewidth(0.6)
    representative_color = CMAP(NORM(45))
    legend = [
        Line2D([], [], color=representative_color, linewidth=HIGH_FORCE_LINEWIDTH,
               label="High-force contact"),
        Line2D([], [], color=representative_color, marker="o",
               markersize=LEGEND_SYMBOL_SIZE, linewidth=0,
               label="Particle with high-force contact"),
        Line2D([], [], color="#999999", linewidth=0.8, alpha=0.5,
               label="Non-high-force contact"),
        Line2D([], [], color="#999999", marker="o",
               markersize=LEGEND_SYMBOL_SIZE, linewidth=0, alpha=0.5,
               label="Particle with no high-force contact"),
    ]
    if include_walls:
        legend.append(Line2D([], [], color="#999999", marker="^",
                             markersize=LEGEND_SYMBOL_SIZE, linewidth=0,
                             label="Wall-contact location"))
    legend_artist = fig.legend(handles=legend, loc="upper center",
               bbox_to_anchor=(0.50, system_fraction.y0 - 0.055),
               frameon=False, ncol=2, fontsize=TEXT_FONTSIZE,
               handlelength=1.3, handletextpad=0.45, columnspacing=1.0,
               labelspacing=0.45, borderpad=0)
    # Direction labels are geometric coordinates, independent of specimen orientation.
    label = {"along_x": "View along x · horizontal y, vertical z",
             "along_y": "View along y · horizontal −x, vertical z",
             "perspective_3d": "3D view",
             "perspective_3d_cylinder": "3D view · cylinder outline"}[view]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend_artist.get_window_extent(renderer)
    legend_fraction = legend_bbox.transformed(fig.transFigure.inverted())
    view_label = fig.text(0.50, legend_fraction.y0 - 0.035, label, ha="center", va="top",
                          fontsize=TEXT_FONTSIZE, color="#555555")
    fig.canvas.draw()
    output_bbox = Bbox.union([
        system_bbox.padded(5), cax.get_tightbbox(renderer),
        legend_artist.get_window_extent(renderer), view_label.get_window_extent(renderer),
    ]).transformed(fig.dpi_scale_trans.inverted()).padded(0.12)
    paths = []
    for extension in formats:
        path = target.with_suffix("." + extension)
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=output_bbox,
                    metadata={"Creator": "ManuscriptVisualization"})
        paths.append(str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-split", choices=("force_split1", "force_split2"), default="force_split2")
    parser.add_argument("--input", type=Path, help="Override labeled graph pickle")
    parser.add_argument("--output-root", type=Path, default=CASE / "Visualization" / "high_force_contacts")
    parser.add_argument("--angles", nargs="+", default=["0deg", "15deg", "30deg", "45deg"])
    parser.add_argument("--samples-per-angle", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--views", nargs="+", choices=tuple(VIEWS), default=list(VIEWS))
    args = parser.parse_args()
    if args.samples_per_angle < 1 or args.dpi < 1:
        parser.error("samples-per-angle and dpi must be positive")
    source = args.input or (CASE / "0_graph_and_basic_stats" / "force_splits" / args.force_split / "graph_dict_labeled.pkl")
    with source.open("rb") as handle:
        graphs = pickle.load(handle)
    threshold_path = source.parent / "high_force_threshold_info.pkl"
    threshold = None
    if threshold_path.exists():
        with threshold_path.open("rb") as handle:
            threshold = pickle.load(handle)
    selections = {}
    for label in args.angles:
        if not re.fullmatch(r"\d+deg", label) or label not in graphs:
            parser.error(f"Unknown orientation: {label}")
        count = len(graphs[label]["full"])
        if count < args.samples_per_angle:
            parser.error(f"{label} has only {count} systems")
        # Separate deterministic streams keep each selection stable when angles are reordered.
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(label[:-3])]))
        selections[label] = sorted(int(i) for i in rng.choice(count, args.samples_per_angle, replace=False))
    root = args.output_root / args.force_split
    manifest = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "force_split": args.force_split, "threshold_info": threshold, "seed": args.seed,
                "selected_list_indices_zero_based": selections, "angle_field": "angle_with_zz",
                "color_limits_deg": [NORM.vmin, NORM.vmax], "colormap": CMAP.name, "dpi": args.dpi,
                "out_of_range_colors": "Clipped to endpoint colors; colorbar extends at both ends",
                "high_force_linewidth_pt": HIGH_FORCE_LINEWIDTH,
                "font_family": "Helvetica with Nimbus Sans/Arial/DejaVu Sans fallbacks",
                "text_fontsize_pt": TEXT_FONTSIZE,
                "legend_symbol_size_pt": LEGEND_SYMBOL_SIZE,
                "views_elevation_azimuth_deg": VIEWS, "projection": "orthographic",
                "node_color_rule": "Angle of strongest visible incident high-force edge; otherwise gray",
                "wall_position_rule": "Recorded contact_location, one node per wall contact",
                "cylinder_rule": "Plain open cylinder fitted to sidewall contacts without radius inflation; outer rings at full height extrema; no end-cap meshes or inner rings; framing zoom 1.45",
                "systems": []}
    for label, indices in selections.items():
        for index in indices:
            graph = graphs[label]["full"][index]
            sim_idx = int(graph.graph.get("sim_idx", index))
            geometry = prepare_geometry(graph)
            for include_walls in (False, True):
                variant = "with_walls" if include_walls else "without_walls"
                output = root / label / f"sim_{sim_idx:03d}" / variant
                output.mkdir(parents=True, exist_ok=True)
                network = visible_network(graph, geometry[0], include_walls)
                entry = {"angle": label, "list_index": index, "sim_idx": sim_idx,
                         "variant": variant, "nodes": len(network[0]), "high_force_contacts": len(network[2]),
                         "other_contacts": len(network[1]), "wall_contacts": network[5],
                         "high_force_wall_contacts": network[6], "files": []}
                if "perspective_3d_cylinder" in args.views:
                    entry["cylinder_envelope"] = cylinder_envelope(graph, geometry)
                for view in args.views:
                    stem = f"final_load__{args.force_split}__{label}__sim_{sim_idx:03d}__high_force_angle_zz__{variant}__{view}"
                    entry["files"].extend(draw_figure(graph, geometry, network, view, include_walls,
                                                    output / stem, args.formats, args.dpi))
                manifest["systems"].append(entry)
            print(f"Finished {label} sim_{sim_idx:03d}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    manifest_name = "selection_and_render_manifest"
    if set(args.views) != set(VIEWS):
        manifest_name += "__" + "__".join(args.views)
    (root / (manifest_name + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {sum(len(e['files']) for e in manifest['systems'])} files in {root}")


if __name__ == "__main__":
    main()
