#!/usr/bin/env python3
"""Plot full crystal systems and primitive unit cells with center equations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "crystal_reference_config.json"
VIEWS = (
    (0, 0, "Projection along x"),
    (0, 90, "Projection along y"),
    (90, -90, "Projection along z"),
    (24, -52, "Perspective 3D"),
)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    root = Path(cfg.get("project_root") or PROJECT_ROOT).resolve()
    cfg["project_root"] = str(root)
    cfg["output_root"] = str((root / cfg["output_root"]).resolve())
    return cfg


def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def minimum_image(delta: np.ndarray, lengths: dict[int, float]) -> np.ndarray:
    result = np.asarray(delta, float).copy()
    for axis in (0, 1):
        result[axis] -= lengths[axis] * np.round(result[axis] / lengths[axis])
    return result


def split_periodic_segment(start: np.ndarray, delta: np.ndarray, lengths: dict[int, float]) -> list[np.ndarray]:
    current = start.copy()
    remaining = delta.copy()
    pieces = []
    for _ in range(3):
        endpoint = current + remaining
        crossings = []
        for axis in (0, 1):
            length = lengths[axis]
            if endpoint[axis] < 0 and remaining[axis] < 0:
                crossings.append(((0 - current[axis]) / remaining[axis], axis, length))
            elif endpoint[axis] > length and remaining[axis] > 0:
                crossings.append(((length - current[axis]) / remaining[axis], axis, 0.0))
        crossings = [item for item in crossings if 0 < item[0] < 1]
        if not crossings:
            pieces.append(np.vstack((current, endpoint)))
            break
        fraction = min(item[0] for item in crossings)
        hit = current + fraction * remaining
        pieces.append(np.vstack((current, hit)))
        remaining = (1 - fraction) * remaining
        current = hit.copy()
        for other_fraction, axis, wrapped in crossings:
            if np.isclose(other_fraction, fraction):
                current[axis] = wrapped
    return pieces


def core_graph(full):
    nodes = [node for node, data in full.nodes(data=True) if not data.get("is_wall", False)]
    return full.subgraph(nodes).copy()


def plot_system(graph, structure: str, output: Path) -> None:
    nodes = list(graph)
    xyz = np.asarray([graph.nodes[node]["position"] for node in nodes], float)
    index = {node: idx for idx, node in enumerate(nodes)}
    cell = np.asarray(graph.graph["cell_matrix"], float)
    lengths = {axis: float(cell[axis, axis]) for axis in range(3)}
    segments = []
    for u, v in graph.edges():
        start = xyz[index[u]]
        delta = minimum_image(xyz[index[v]] - start, lengths)
        segments.extend(split_periodic_segment(start, delta, lengths))
    bottom = np.asarray([bool(graph.nodes[node].get("contacts_bottom_wall", False)) for node in nodes])
    top = np.asarray([bool(graph.nodes[node].get("contacts_top_wall", False)) for node in nodes])
    interior = ~(bottom | top)
    mins = np.array([0.0, 0.0, xyz[:, 2].min()])
    maxs = np.array([lengths[0], lengths[1], xyz[:, 2].max()])
    span = np.maximum(maxs - mins, np.finfo(float).eps)
    fig = plt.figure(figsize=(12, 10))
    axes = [fig.add_subplot(2, 2, idx + 1, projection="3d") for idx in range(4)]
    for axis, (elev, azim, title) in zip(axes, VIEWS):
        axis.add_collection3d(
            Line3DCollection(segments, colors="#7A9CC6", linewidths=0.28, alpha=0.30, rasterized=True)
        )
        axis.scatter(xyz[interior, 0], xyz[interior, 1], xyz[interior, 2], s=2.6, color="#6B6B6B", alpha=0.55, depthshade=False, rasterized=True)
        axis.scatter(xyz[bottom, 0], xyz[bottom, 1], xyz[bottom, 2], s=7, color="#1F77B4", alpha=0.95, depthshade=False, rasterized=True)
        axis.scatter(xyz[top, 0], xyz[top, 1], xyz[top, 2], s=7, color="#FF7F0E", alpha=0.95, depthshade=False, rasterized=True)
        axis.view_init(elev=elev, azim=azim)
        axis.set_proj_type("persp" if title == "Perspective 3D" else "ortho")
        axis.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]), zlim=(mins[2], maxs[2]), title=title)
        axis.set_box_aspect(span)
        axis.set_axis_off()
    fig.suptitle(f"{structure} crystal reference — full system, 0° orientation, periodic x/y")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color="#6B6B6B", label="Interior particle"),
            Line2D([0], [0], marker="o", linestyle="", color="#1F77B4", label="Bottom-surface particle"),
            Line2D([0], [0], marker="o", linestyle="", color="#FF7F0E", label="Top-surface particle"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.075, top=0.92, wspace=0.02, hspace=0.05)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def lattice_spec(structure: str, diameter: float) -> dict:
    d = float(diameter)
    if structure == "SC":
        a = d
        vectors = np.asarray([[a, 0, 0], [0, a, 0], [0, 0, a]], float)
        basis = np.asarray([[0, 0, 0]], float)
        symbolic = (
            r"$a=d$",
            r"$a_1=a(1,0,0)$",
            r"$a_2=a(0,1,0)$",
            r"$a_3=a(0,0,1)$",
            r"$b_1=(0,0,0)$",
        )
        conventional = "SC conventional cell = primitive cell"
    elif structure == "BCC":
        a = 2 * d / np.sqrt(3)
        vectors = 0.5 * a * np.asarray([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
        basis = np.asarray([[0, 0, 0]], float)
        symbolic = (
            r"$a_c=2d/\sqrt{3}$",
            r"$a_1=\frac{a_c}{2}(-1,1,1)$",
            r"$a_2=\frac{a_c}{2}(1,-1,1)$",
            r"$a_3=\frac{a_c}{2}(1,1,-1)$",
            r"$b_1=(0,0,0)$",
        )
        conventional = r"Conventional basis: $(0,0,0),(1/2,1/2,1/2)$"
    elif structure == "FCC":
        a = np.sqrt(2) * d
        vectors = 0.5 * a * np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
        basis = np.asarray([[0, 0, 0]], float)
        symbolic = (
            r"$a_c=\sqrt{2}d$",
            r"$a_1=\frac{a_c}{2}(0,1,1)$",
            r"$a_2=\frac{a_c}{2}(1,0,1)$",
            r"$a_3=\frac{a_c}{2}(1,1,0)$",
            r"$b_1=(0,0,0)$",
        )
        conventional = r"Conventional basis: $(0,0,0),(0,1/2,1/2),(1/2,0,1/2),(1/2,1/2,0)$"
    elif structure == "HCP":
        c = np.sqrt(8 / 3) * d
        vectors = np.asarray([[d, 0, 0], [0.5 * d, np.sqrt(3) * d / 2, 0], [0, 0, c]], float)
        basis = np.asarray([[0, 0, 0], [1 / 3, 1 / 3, 1 / 2]], float)
        symbolic = (
            r"$a=d,\quad c=\sqrt{8/3}d$",
            r"$a_1=a(1,0,0)$",
            r"$a_2=a(1/2,\sqrt{3}/2,0)$",
            r"$a_3=c(0,0,1)$",
            r"$b_1=(0,0,0),\quad b_2=(1/3,1/3,1/2)$",
        )
        conventional = "Two-particle primitive basis; ideal c/a ratio"
    else:
        raise ValueError(structure)
    return {
        "structure": structure,
        "vectors": vectors,
        "basis_fractional": basis,
        "symbolic": symbolic,
        "note": conventional,
        "lattice_parameter": a if structure != "HCP" else d,
    }


def cell_vertices(vectors: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    fractions = np.asarray(list(itertools.product((0, 1), repeat=3)), float)
    vertices = fractions @ vectors
    edges = []
    for i, left in enumerate(fractions):
        for j in range(i + 1, len(fractions)):
            right = fractions[j]
            if np.sum(np.abs(left - right)) == 1:
                edges.append((i, j))
    return vertices, edges


def displayed_basis_points(vectors: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fractional = []
    labels = []
    for basis_id, center in enumerate(basis):
        for shift in itertools.product((0, 1), repeat=3):
            point = center + np.asarray(shift, float)
            if np.all(point >= -1e-12) and np.all(point <= 1 + 1e-12):
                fractional.append(point)
                labels.append(basis_id)
    points = np.asarray(fractional) @ vectors
    rounded = np.round(points, 14)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    unique = np.sort(unique)
    return points[unique], np.asarray(labels)[unique]


def plot_unit_cell(spec: dict, diameter: float, output: Path) -> None:
    vectors = spec["vectors"]
    basis = spec["basis_fractional"]
    vertices, edge_pairs = cell_vertices(vectors)
    centers, basis_ids = displayed_basis_points(vectors, basis)
    segments = [np.vstack((vertices[i], vertices[j])) for i, j in edge_pairs]
    # Contact-sized segments among displayed centers make the local coordination
    # visible without treating the gray cell frame as a contact network.
    contacts = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if np.isclose(np.linalg.norm(centers[j] - centers[i]), diameter, rtol=0.03, atol=diameter * 0.01):
                contacts.append(np.vstack((centers[i], centers[j])))
    fig = plt.figure(figsize=(12, 6.2))
    axis = fig.add_subplot(1, 2, 1, projection="3d")
    axis.add_collection3d(Line3DCollection(segments, colors="#333333", linewidths=1.3, alpha=0.85))
    if contacts:
        axis.add_collection3d(Line3DCollection(contacts, colors="#4C78A8", linewidths=1.8, alpha=0.72))
    palette = ["#D62728", "#2CA02C"]
    for basis_id in sorted(set(basis_ids)):
        mask = basis_ids == basis_id
        axis.scatter(
            centers[mask, 0], centers[mask, 1], centers[mask, 2],
            s=75, color=palette[basis_id % len(palette)], edgecolor="white", linewidth=0.6,
            label=f"Basis {basis_id + 1}", depthshade=False,
        )
    origin = np.zeros(3)
    for idx, vector in enumerate(vectors, start=1):
        axis.quiver(*origin, *vector, color="#111111", arrow_length_ratio=0.12, linewidth=1.2)
        endpoint = vector * 1.08
        axis.text(*endpoint, f"a{idx}", fontsize=10)
    combined = np.vstack((vertices, centers))
    spans = np.ptp(combined, axis=0)
    spans[spans == 0] = diameter
    axis.set_box_aspect(spans)
    axis.set_title(f"{spec['structure']} primitive unit cell")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.legend(frameon=False)
    text_axis = fig.add_subplot(1, 2, 2)
    text_axis.axis("off")
    general = r"$r_{n_1n_2n_3,s}=n_1a_1+n_2a_2+n_3a_3+b_s$"
    integer_note = r"$n_1,n_2,n_3\in Z$"
    lines = [
        f"Particle diameter: d = {diameter:.6g} m",
        "",
        *spec["symbolic"],
        "",
        "Particle-center equation:",
        general,
        integer_note,
        "",
        spec["note"],
        "",
        "Numeric primitive vectors (m):",
        *[f"a{idx + 1} = ({vector[0]:.8g}, {vector[1]:.8g}, {vector[2]:.8g})" for idx, vector in enumerate(vectors)],
    ]
    text_axis.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=11, linespacing=1.45)
    fig.suptitle(f"{spec['structure']}: smallest repeating unit and particle centers")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_equation_files(specs: list[dict], diameter: float, output_root: Path) -> None:
    rows = []
    markdown = [
        "# Crystal lattice vectors and particle-center equations",
        "",
        "For every structure, particle centers follow",
        "",
        r"$$r_{n_1n_2n_3,s}=n_1a_1+n_2a_2+n_3a_3+b_s,\qquad n_i\in\mathbb Z.$$",
        "",
        f"The nearest-neighbor particle diameter is `d = {diameter:.8g} m`.",
        "",
    ]
    for spec in specs:
        markdown.extend([f"## {spec['structure']}", "", *spec["symbolic"], "", spec["note"], ""])
        for vector_id, vector in enumerate(spec["vectors"], start=1):
            rows.append(
                {
                    "structure": spec["structure"],
                    "record": f"a{vector_id}",
                    "coordinate_type": "cartesian_m",
                    "x": vector[0],
                    "y": vector[1],
                    "z": vector[2],
                }
            )
        for basis_id, basis in enumerate(spec["basis_fractional"], start=1):
            rows.append(
                {
                    "structure": spec["structure"],
                    "record": f"b{basis_id}",
                    "coordinate_type": "fractional_in_primitive_vectors",
                    "x": basis[0],
                    "y": basis[1],
                    "z": basis[2],
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "particle_center_equations.md").write_text("\n".join(markdown) + "\n")
    with (output_root / "particle_center_equations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["structure", "record", "coordinate_type", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "README.md").write_text(
        "# Crystal geometry figures\n\n"
        "`system_views/` contains the full approximately 1,500-particle system in "
        "x/y/z projections and 3D perspective. `unit_cells/` contains the primitive "
        "unit cell, lattice vectors, basis centers, and particle-center equation for "
        "SC, BCC, FCC, and HCP.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    output_root = Path(cfg["output_root"]) / "1_network_property_comparison" / "sample_systems"
    raw_root = Path(cfg["output_root"]) / "0_graph_and_basic_stats" / "artifacts" / "raw_graphs"
    diameter = float(cfg["particle_diameter"])
    specs = []
    for structure in cfg["structures"]:
        full = load_graph(raw_root / f"{structure}.pkl")
        plot_system(
            core_graph(full),
            structure,
            output_root / "system_views" / f"{structure}_system_four_views.png",
        )
        spec = lattice_spec(structure, diameter)
        specs.append(spec)
        plot_unit_cell(
            spec,
            diameter,
            output_root / "unit_cells" / f"{structure}_primitive_unit_and_equations.png",
        )
    write_equation_files(specs, diameter, output_root)
    print(f"Created {len(cfg['structures'])} system views and {len(specs)} primitive-cell figures in {output_root}")


if __name__ == "__main__":
    main()
