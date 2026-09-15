#!/usr/bin/env python3
"""Reconstruct trusted NetworkX graph views from graph-construction CSV tables.

This is useful when feature CSVs are available but an externally produced
pickle should not be executed.  The reconstructed pickle is written locally by
this script and is therefore suitable for the downstream analysis pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ID_COLUMNS = {"geometry", "sim_idx", "node_id", "node1", "node2", "is_core_edge"}
VECTOR_COLUMNS = {
    "position": ("x", "y", "z"),
    "contact_location": ("contact_x", "contact_y", "contact_z"),
    "n_unit": ("n_x", "n_y", "n_z"),
    "t_unit": ("t_x", "t_y", "t_z"),
}


def scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def node_id(value):
    """Restore numeric particle IDs while retaining string wall IDs."""
    value = scalar(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return int(float(text))
    return text


def attributes(row, excluded, vector_names=()):
    excluded = set(excluded)
    for name in vector_names:
        excluded.update(VECTOR_COLUMNS[name])
    result = {}
    for key, value in row.items():
        if key in excluded:
            continue
        value = scalar(value)
        if value is not None:
            result[key] = value
    for name in vector_names:
        columns = VECTOR_COLUMNS[name]
        values = tuple(scalar(row.get(column)) for column in columns)
        if all(value is not None for value in values):
            result[name] = tuple(float(value) for value in values)
    return result


def graph_attributes(graph_row):
    attrs = attributes(graph_row, {"geometry", "sim_idx"})
    attrs["angle_label"] = str(graph_row["geometry"])
    attrs["sim_idx"] = int(graph_row["sim_idx"])
    return attrs


def reconstruct(source: Path):
    nodes = pd.read_csv(source / "node_features.csv")
    edges = pd.read_csv(source / "edge_features.csv", low_memory=False)
    graph_rows = pd.read_csv(source / "graph_features.csv")

    required_nodes = {"geometry", "sim_idx", "node_id", "x", "y", "z"}
    required_edges = {"geometry", "sim_idx", "node1", "node2", "is_core_edge"}
    if missing := required_nodes.difference(nodes.columns):
        raise KeyError(f"node_features.csv is missing {sorted(missing)}")
    if missing := required_edges.difference(edges.columns):
        raise KeyError(f"edge_features.csv is missing {sorted(missing)}")

    graph_lookup = {
        (str(row.geometry), int(row.sim_idx)): row._asdict()
        for row in graph_rows.itertuples(index=False)
    }
    result = {}
    audits = []
    ordered_keys = list(dict.fromkeys(zip(nodes["geometry"].astype(str), nodes["sim_idx"].astype(int))))
    for geometry, sim_idx in ordered_keys:
        node_frame = nodes[(nodes.geometry.astype(str) == geometry) & (nodes.sim_idx == sim_idx)]
        edge_frame = edges[(edges.geometry.astype(str) == geometry) & (edges.sim_idx == sim_idx)]
        key = (geometry, sim_idx)
        if key not in graph_lookup:
            raise KeyError(f"Missing graph_features.csv row for {geometry} simulation {sim_idx}")
        graph_row = graph_lookup[key]
        attrs = graph_attributes(graph_row)

        full = nx.Graph(**attrs)
        core = nx.Graph(**attrs)
        for row in node_frame.to_dict("records"):
            identifier = node_id(row["node_id"])
            data = attributes(row, ID_COLUMNS, ("position",))
            data["is_wall"] = bool(data.get("is_wall", False))
            full.add_node(identifier, **data)
            core.add_node(identifier, **data)

        for row in edge_frame.to_dict("records"):
            u, v = node_id(row["node1"]), node_id(row["node2"])
            data = attributes(row, ID_COLUMNS, ("contact_location", "n_unit", "t_unit"))
            is_core = bool(row["is_core_edge"])
            for identifier in (u, v):
                if identifier not in full:
                    full.add_node(identifier, is_wall=True)
            full.add_edge(u, v, **data)
            if is_core:
                if u not in core or v not in core:
                    raise ValueError(f"Core edge {u!r}-{v!r} references a wall/missing node")
                core.add_edge(u, v, **data)

        expected_core_nodes = int(graph_row.get("num_nodes", len(node_frame)))
        expected_full_nodes = int(graph_row.get("num_nodes_with_walls", full.number_of_nodes()))
        expected_core_edges = int(graph_row.get("num_edges", int(edge_frame.is_core_edge.sum())))
        expected_full_edges = int(graph_row.get("num_edges_with_walls", len(edge_frame)))
        observed = (core.number_of_nodes(), full.number_of_nodes(), core.number_of_edges(), full.number_of_edges())
        expected = (expected_core_nodes, expected_full_nodes, expected_core_edges, expected_full_edges)
        if observed != expected:
            raise ValueError(f"{geometry} simulation {sim_idx}: observed {observed}, expected {expected}")

        degree_mismatches = 0
        if "degree" in node_frame:
            stored_degree = {node_id(row.node_id): int(row.degree) for row in node_frame.itertuples()}
            degree_mismatches = sum(core.degree(identifier) != degree for identifier, degree in stored_degree.items())
            if degree_mismatches:
                raise ValueError(f"{geometry} simulation {sim_idx}: {degree_mismatches} core-degree mismatches")

        result.setdefault(geometry, {"full": [], "core": []})
        result[geometry]["full"].append(full)
        result[geometry]["core"].append(core)
        audits.append({
            "geometry": geometry,
            "sim_idx": sim_idx,
            "core_nodes": core.number_of_nodes(),
            "full_nodes": full.number_of_nodes(),
            "core_edges": core.number_of_edges(),
            "full_edges": full.number_of_edges(),
            "degree_mismatches": degree_mismatches,
        })
    return result, pd.DataFrame(audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    graph_path = output / "graph_dict_labeled.pkl"
    if graph_path.exists() and not args.overwrite:
        raise FileExistsError(f"{graph_path} exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    graph_dict, audit = reconstruct(source)
    with graph_path.open("wb") as handle:
        pickle.dump(graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    audit.to_csv(output / "csv_graph_reconstruction_audit.csv", index=False)

    for name in (
        "node_features.csv", "edge_features.csv", "graph_features.csv",
        "graph_feature_arrays.csv", "high_force_edges.csv",
        "pair_edge_connectivity_index.csv", "simulation_slices.txt",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, output / name)

    metadata = {
        "source": str(source),
        "pickle_source": "reconstructed_from_CSV_without_loading_external_pickle",
        "systems": int(len(audit)),
        "geometries": list(graph_dict),
        "core_nodes": int(audit.core_nodes.sum()),
        "core_edges": int(audit.core_edges.sum()),
        "full_nodes": int(audit.full_nodes.sum()),
        "full_edges": int(audit.full_edges.sum()),
        "degree_mismatches": int(audit.degree_mismatches.sum()),
    }
    (output / "csv_graph_reconstruction_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Jamming graph data\n\n"
        "The local graph pickle was reconstructed from the Turbo `node_features.csv`, "
        "`edge_features.csv`, and `graph_features.csv` tables without loading the "
        "externally produced pickle. Counts and particle degrees were validated; see "
        "`csv_graph_reconstruction_audit.csv` and "
        "`csv_graph_reconstruction_metadata.json`.\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
