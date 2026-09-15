#!/usr/bin/env python3
"""Create a separate graph dataset with exact periodic geometry attributes.

The original graph topology and attributes are retained. Particle-particle
edges receive minimum-image displacement/distance fields, and the canonical
``angle_with_zz`` is recomputed from that displacement. Wall edges receive the
raw negative wall label and a top/bottom surface classification.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT / "AnalysisScripts" / "Pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from GraphPipelineCommon import _build_feature_tables  # noqa: E402


DEFAULT_INPUT = (
    PROJECT
    / "AnalysisResults"
    / "periodic_boundaries"
    / "2026-08-03"
    / "_archive_pre_periodic_correction"
    / "graph_features"
    / "graph_dict_labeled.pkl"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "AnalysisResults"
    / "periodic_boundaries"
    / "2026-08-03"
    / "0_graph_and_basic_stats"
    / "graph_data"
)
DEFAULT_BOX_LENGTHS = (0.0018, 0.0030)
PERIODIC_AXES = (0, 1)
PARTICLE_DIAMETER = 0.00015
BOTTOM_WALL_LABELS = {-1, -2, -3}
TOP_WALL_LABELS = {-4, -5}


def wall_surface(label: int | None) -> str:
    if label in BOTTOM_WALL_LABELS:
        return "bottom"
    if label in TOP_WALL_LABELS:
        return "top"
    return "unknown"


def periodic_edge_attributes(position_u, position_v, box_lengths):
    raw = np.asarray(position_v, float) - np.asarray(position_u, float)
    displacement = raw.copy()
    shifts = np.zeros(3, dtype=int)
    for axis, length in zip(PERIODIC_AXES, box_lengths):
        shifts[axis] = -int(np.round(displacement[axis] / length))
        displacement[axis] += shifts[axis] * length
    distance = float(np.linalg.norm(displacement))
    angle = (
        float(np.degrees(np.arccos(np.clip(abs(displacement[2]) / distance, 0.0, 1.0))))
        if distance > 0
        else np.nan
    )
    return raw, displacement, shifts, distance, angle


def correct_graphs(graph_dict, box_lengths):
    audit = Counter()
    distance_errors = []
    angle_differences = []

    for geometry, views in graph_dict.items():
        for sim_idx, (full, core) in enumerate(zip(views["full"], views["core"])):
            graph_metadata = {
                "box_lengths": {0: box_lengths[0], 1: box_lengths[1]},
                "periodic_axes": list(PERIODIC_AXES),
                "periodic_geometry_source": "exact_from_raw_contact_normals_and_overlap",
                "periodic_geometry_corrected": True,
            }
            full.graph.update(graph_metadata)
            core.graph.update(graph_metadata)

            for u, v, data in full.edges(data=True):
                u_wall = bool(full.nodes[u].get("is_wall", False))
                v_wall = bool(full.nodes[v].get("is_wall", False))
                if u_wall or v_wall:
                    wall_node = u if u_wall else v
                    label = int(full.nodes[wall_node].get("wall_label"))
                    data["wall_label"] = label
                    data["wall_surface"] = wall_surface(label)
                    audit[f"wall_{data['wall_surface']}"] += 1
                    continue

                raw, displacement, shifts, distance, angle = periodic_edge_attributes(
                    full.nodes[u]["position"], full.nodes[v]["position"], box_lengths
                )
                original_angle = float(data.get("angle_with_zz", np.nan))
                data.update(
                    angle_with_zz_original=original_angle,
                    angle_with_zz=angle,
                    angle_with_zz_periodic=angle,
                    periodic_dx=float(displacement[0]),
                    periodic_dy=float(displacement[1]),
                    periodic_dz=float(displacement[2]),
                    periodic_distance=distance,
                    periodic_shift_x=int(shifts[0]),
                    periodic_shift_y=int(shifts[1]),
                    crosses_periodic_x=bool(shifts[0]),
                    crosses_periodic_y=bool(shifts[1]),
                    is_periodic_crossing=bool(shifts[0] or shifts[1]),
                )
                audit["particle_edges"] += 1
                audit["periodic_particle_edges"] += int(bool(shifts[0] or shifts[1]))
                expected_distance = PARTICLE_DIAMETER - float(data.get("delta", 0.0))
                distance_errors.append(abs(distance - expected_distance))
                if np.isfinite(original_angle):
                    angle_differences.append(abs(angle - original_angle))

                if core.has_edge(u, v):
                    core[u][v].update(data)

            audit["simulations"] += 1
            audit[f"simulations_{geometry}"] += 1

    validation = {
        **{key: int(value) for key, value in sorted(audit.items())},
        "particle_diameter_m": PARTICLE_DIAMETER,
        "box_lengths_m": {"0": box_lengths[0], "1": box_lengths[1]},
        "periodic_axes": list(PERIODIC_AXES),
        "maximum_contact_distance_residual_m": float(max(distance_errors, default=np.nan)),
        "median_contact_distance_residual_m": float(np.median(distance_errors)) if distance_errors else np.nan,
        "maximum_angle_difference_from_raw_normal_deg": float(max(angle_differences, default=np.nan)),
        "median_angle_difference_from_raw_normal_deg": float(np.median(angle_differences)) if angle_differences else np.nan,
    }
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lx", type=float, default=DEFAULT_BOX_LENGTHS[0])
    parser.add_argument("--ly", type=float, default=DEFAULT_BOX_LENGTHS[1])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open("rb") as handle:
        graph_dict = pickle.load(handle)

    validation = correct_graphs(graph_dict, (args.lx, args.ly))
    output_pickle = args.output_dir / "graph_dict_labeled.pkl"
    with output_pickle.open("wb") as handle:
        pickle.dump(graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    nodes, edges, graphs, _ = _build_feature_tables(graph_dict)
    nodes.to_csv(args.output_dir / "node_features.csv", index=False)
    edges.to_csv(args.output_dir / "edge_features.csv", index=False)
    graphs.to_csv(args.output_dir / "graph_features.csv", index=False)

    threshold_source = args.input.parent / "high_force_threshold_info.pkl"
    if threshold_source.exists():
        shutil.copy2(threshold_source, args.output_dir / threshold_source.name)

    metadata = {
        "source_graph_pickle": str(args.input),
        "output_graph_pickle": str(output_pickle),
        "original_results_preserved": True,
        "canonical_contact_angle": "angle_with_zz recomputed from exact minimum-image particle-center displacement",
        "raw_normal_angle_backup": "angle_with_zz_original",
        "wall_surface_mapping": {
            "bottom": sorted(BOTTOM_WALL_LABELS),
            "top": sorted(TOP_WALL_LABELS),
        },
        "validation": validation,
    }
    (args.output_dir / "periodic_geometry_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    print(f"Saved corrected graph dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
