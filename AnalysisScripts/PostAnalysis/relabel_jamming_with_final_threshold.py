#!/usr/bin/env python
"""Relabel jamming-state high-force contacts with the final-state threshold."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_SCRIPTS_DIR = SCRIPT_DIR.parent
DEFAULT_FINAL_DIR = (
    ANALYSIS_SCRIPTS_DIR.parent
    / "AnalysisResults"
    / "FinalLoadState"
    / "FullGraph_2mean_ref_geom"
)
DEFAULT_JAMMING_DIR = (
    ANALYSIS_SCRIPTS_DIR.parent
    / "AnalysisResults"
    / "JammingState"
    / "FullGraph_2mean_ref_geom"
)
DEFAULT_OUT_DIR = (
    ANALYSIS_SCRIPTS_DIR.parent
    / "AnalysisResults"
    / "JammingState"
    / "FullGraph_final_load_threshold"
)


def serialize_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def flatten_named_vector(record: dict, prefix: str, value, names: tuple[str, ...]) -> bool:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if arr.size != len(names):
        return False
    for name, item in zip(names, arr):
        record[name] = float(item)
    return True


def graph_id_fields(G) -> dict:
    return {
        "geometry": G.graph.get("angle_label", G.graph.get("geometry")),
        "sim_idx": G.graph.get("sim_idx"),
    }


def tag_high_force_edges_and_nodes(G_full, G_core, threshold: float) -> list[dict]:
    edge_records = []
    for u, v, data in G_full.edges(data=True):
        data["is_high_force"] = data.get("normal_force", 0.0) >= threshold

    high_force_degree = {}
    for u, v, data in G_full.edges(data=True):
        if not data["is_high_force"]:
            continue
        if not G_full.nodes[u].get("is_wall", False):
            high_force_degree[u] = high_force_degree.get(u, 0) + 1
        if not G_full.nodes[v].get("is_wall", False):
            high_force_degree[v] = high_force_degree.get(v, 0) + 1

        def map_node(node):
            return node if not G_full.nodes[node].get("is_wall", False) else -1

        edge_records.append(
            {
                "geometry": G_full.graph.get("angle_label"),
                "sim_idx": G_full.graph.get("sim_idx"),
                "node1": map_node(u),
                "node2": map_node(v),
            }
        )

    for node in G_full.nodes:
        if G_full.nodes[node].get("is_wall", False):
            continue
        degree = high_force_degree.get(node, 0)
        G_full.nodes[node]["high_force_degree"] = degree
        G_full.nodes[node]["is_force_chain_node"] = degree >= 1
        G_full.nodes[node]["force_chain_role"] = (
            "none" if degree == 0 else "endpoint" if degree == 1 else "transmission"
        )

    for u, v in G_core.edges:
        if G_full.has_edge(u, v):
            G_core[u][v]["is_high_force"] = G_full[u][v].get("is_high_force", False)
    for node in G_core.nodes:
        degree = high_force_degree.get(node, 0)
        G_core.nodes[node]["high_force_degree"] = degree
        G_core.nodes[node]["is_force_chain_node"] = degree >= 1
        G_core.nodes[node]["force_chain_role"] = (
            "none" if degree == 0 else "endpoint" if degree == 1 else "transmission"
        )

    return edge_records


def build_feature_tables(graph_dict: dict):
    node_records = []
    edge_records = []
    graph_records = []
    graph_array_records = []

    for graphs in graph_dict.values():
        for G_full, G_core in zip(graphs["full"], graphs["core"]):
            graph_id = graph_id_fields(G_full)

            graph_row = dict(graph_id)
            graph_array_row = dict(graph_id)
            for key, value in G_full.graph.items():
                if key in {"angle_label", "sim_idx"}:
                    continue
                value = serialize_value(value)
                if isinstance(value, (list, tuple, dict)):
                    graph_array_row[key] = value
                else:
                    graph_row[key] = value
            graph_records.append(graph_row)
            if len(graph_array_row) > len(graph_id):
                graph_array_records.append(graph_array_row)

            for node, data in G_core.nodes(data=True):
                record = {**graph_id, "node_id": node}
                for key, value in data.items():
                    if key == "position" and flatten_named_vector(record, key, value, ("x", "y", "z")):
                        continue
                    record[key] = serialize_value(value)
                node_records.append(record)

            for u, v, data in G_full.edges(data=True):
                record = {**graph_id, "node1": u, "node2": v, "is_core_edge": bool(G_core.has_edge(u, v))}
                for key, value in data.items():
                    if key == "contact_location" and flatten_named_vector(
                        record, key, value, ("contact_x", "contact_y", "contact_z")
                    ):
                        continue
                    if key == "n_unit" and flatten_named_vector(record, key, value, ("n_x", "n_y", "n_z")):
                        continue
                    if key == "t_unit" and flatten_named_vector(record, key, value, ("t_x", "t_y", "t_z")):
                        continue
                    record[key] = serialize_value(value)
                edge_records.append(record)

    return (
        pd.DataFrame(node_records),
        pd.DataFrame(edge_records),
        pd.DataFrame(graph_records),
        pd.DataFrame(graph_array_records),
    )


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_final_threshold(final_dir: Path) -> float:
    info = load_pickle(final_dir / "high_force_threshold_info.pkl")
    thresholds = info.get("thresholds", {})
    if "0deg" not in thresholds:
        raise KeyError(f"Could not find final-state 0deg threshold in {final_dir}")
    return float(thresholds["0deg"])


def relabel_graph_dict(graph_dict: dict, threshold: float):
    high_force_records = []
    for label, views in graph_dict.items():
        full_graphs = views.get("full", [])
        core_graphs = views.get("core", [])
        if len(full_graphs) != len(core_graphs):
            raise ValueError(f"{label}: full/core graph counts do not match")
        for G_full, G_core in zip(full_graphs, core_graphs):
            high_force_records.extend(tag_high_force_edges_and_nodes(G_full, G_core, threshold))
    threshold_info = {
        "mode": "external_reference[FinalLoadState/0deg]",
        "thresholds": {"0deg": threshold},
        "source": str(DEFAULT_FINAL_DIR),
    }
    return high_force_records, threshold_info


def write_feature_tables(graph_dict: dict, out_dir: Path) -> None:
    df_nodes, df_edges, df_graphs, df_graph_arrays = build_feature_tables(graph_dict)
    df_nodes.to_csv(out_dir / "node_features.csv", index=False)
    df_edges.to_csv(out_dir / "edge_features.csv", index=False)
    df_graphs.to_csv(out_dir / "graph_features.csv", index=False)

    df_graph_arrays_serialized = df_graph_arrays.copy()
    for col in df_graph_arrays_serialized.columns:
        if col not in {"geometry", "sim_idx"}:
            df_graph_arrays_serialized[col] = df_graph_arrays_serialized[col].map(serialize_value)
    df_graph_arrays_serialized.to_csv(out_dir / "graph_feature_arrays.csv", index=False)
    df_graph_arrays.to_pickle(out_dir / "graph_feature_arrays.pkl")


def copy_context_files(src_dir: Path, out_dir: Path) -> None:
    for name in [
        "simulation_slices.txt",
        "pair_edge_connectivity_index.csv",
        "mcb_loop_sizes.pkl",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def write_readme(out_dir: Path, final_dir: Path, jamming_dir: Path, threshold: float) -> None:
    text = f"""# JammingState with final-load high-force threshold

This folder was generated from:

- Jamming source: `{jamming_dir}`
- Final-load threshold source: `{final_dir / "high_force_threshold_info.pkl"}`

High-force contacts were relabeled using the final-load `0deg` threshold:

```text
threshold = {threshold}
```

Cluster files are not copied here because cluster membership should be recomputed
after changing the high-force threshold.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--jamming-dir", type=Path, default=DEFAULT_JAMMING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    final_dir = args.final_dir.resolve()
    jamming_dir = args.jamming_dir.resolve()
    out_dir = args.out_dir.resolve()

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists; use --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = load_final_threshold(final_dir)
    graph_dict = load_pickle(jamming_dir / "graph_dict_labeled.pkl")
    high_force_records, threshold_info = relabel_graph_dict(graph_dict, threshold)

    save_pickle(graph_dict, out_dir / "graph_dict_labeled.pkl")
    save_pickle(threshold_info, out_dir / "high_force_threshold_info.pkl")
    pd.DataFrame(high_force_records, columns=["geometry", "sim_idx", "node1", "node2"]).to_csv(
        out_dir / "high_force_edges.csv",
        index=False,
    )
    write_feature_tables(graph_dict, out_dir)
    copy_context_files(jamming_dir, out_dir)
    write_readme(out_dir, final_dir, jamming_dir, threshold)

    print(f"Wrote corrected jamming outputs to: {out_dir}")
    print(f"Final-load threshold applied: {threshold}")
    print(f"High-force edge records: {len(high_force_records)}")


if __name__ == "__main__":
    main()
