#!/usr/bin/env python3
"""Export final-load split-2 particle-only populations into a two-sheet workbook."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
CASE = PROJECT / "AnalysisResults" / "final_load"
SOURCE = CASE / "0_graph_and_basic_stats/force_splits/force_split2"
OUTPUT = CASE / "2_force_cluster_comparison/force_split2/final_load_force_split2_populations.xlsx"
GEOMETRIES = ("0deg", "15deg", "30deg", "45deg")
NODE_PROPERTIES = ("degree", "closeness", "betweenness")


def load_population(scope):
    path = SOURCE / f"{scope}_features.csv"
    frame = pd.read_csv(path)
    excluded = "is_wall" if scope == "node" else "is_wall_contact"
    if frame[excluded].isna().any() or not frame[excluded].isin([True, False]).all():
        raise ValueError(f"Invalid wall flags in {path}")
    frame = frame.loc[~frame[excluded]].copy()
    expected = {(geometry, sim_idx) for geometry in GEOMETRIES for sim_idx in range(20)}
    if set(zip(frame.geometry, frame.sim_idx)) != expected:
        raise ValueError(f"Unexpected geometry/simulation inventory in {path}")
    if scope == "edge":
        if not frame.is_core_edge.all():
            raise ValueError("Non-core edge in particle-only population")
        frame = frame.rename(columns={"node1": "edge_u", "node2": "edge_v"})
        frame["edge_u"] = pd.to_numeric(frame.edge_u, errors="raise").astype(int)
        frame["edge_v"] = pd.to_numeric(frame.edge_v, errors="raise").astype(int)
    return frame, [{"path": str(path.relative_to(PROJECT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]


def export(output):
    node_source, node_files = load_population("node")
    edge_source, edge_files = load_population("edge")
    nodes = node_source[["geometry", "sim_idx", "node_id", *NODE_PROPERTIES]].copy()
    edges = edge_source[["geometry", "sim_idx", "edge_u", "edge_v", "is_high_force",
                         "angle_with_zz", "normal_force"]].copy()
    if edges.is_high_force.isna().any() or not edges.is_high_force.isin([True, False]).all():
        raise ValueError("Invalid high-force edge flags")
    edges.insert(5, "group", np.where(edges.is_high_force, "high_force", "non_high_force"))
    node_keys = ["geometry", "sim_idx", "node_id"]
    strong = edges.loc[edges.is_high_force]
    incident = pd.concat([
        strong[["geometry", "sim_idx", endpoint]].rename(columns={endpoint: "node_id"})
        for endpoint in ("edge_u", "edge_v")
    ], ignore_index=True).groupby(node_keys).size().rename("high_force_degree")
    nodes = nodes.merge(incident, on=node_keys, how="left", validate="one_to_one")
    nodes["high_force_degree"] = nodes.high_force_degree.fillna(0).astype(int)
    nodes.insert(3, "is_force_chain_node", nodes.high_force_degree.gt(0))
    nodes.insert(4, "group", np.where(nodes.is_force_chain_node, "high_force", "non_high_force"))
    stored_labels = node_source.set_index(node_keys).is_force_chain_node
    stored_only_count = int((stored_labels & ~nodes.set_index(node_keys).is_force_chain_node).sum())
    if nodes.duplicated(node_keys).any():
        raise ValueError("Duplicate particle identifiers")
    if edges.duplicated(["geometry", "sim_idx", "edge_u", "edge_v"]).any():
        raise ValueError("Duplicate contact identifiers")
    for endpoint in ("u", "v"):
        lookup = nodes[node_keys + ["is_force_chain_node", *NODE_PROPERTIES]].rename(
            columns={"node_id": f"edge_{endpoint}",
                     **{name: f"{endpoint}_{name}" for name in ["is_force_chain_node", *NODE_PROPERTIES]}})
        edges = edges.merge(lookup, on=["geometry", "sim_idx", f"edge_{endpoint}"],
                            how="left", validate="many_to_one", indicator=True)
        if not edges._merge.eq("both").all():
            raise ValueError("Contact endpoint missing from particle population")
        edges = edges.drop(columns="_merge")
    reference_path = SOURCE / "edge_features.csv"
    threshold = float(2 * edges.loc[edges.geometry.eq("0deg"), "normal_force"].mean())
    if not np.array_equal(edges.normal_force.ge(threshold), edges.is_high_force):
        raise ValueError("Contact labels disagree with the split-2 threshold")
    if not np.isfinite(nodes[list(NODE_PROPERTIES)].to_numpy(dtype=float)).all():
        raise ValueError("Missing particle topology properties")
    if int(nodes.high_force_degree.sum()) != 2 * int(edges.is_high_force.sum()):
        raise ValueError("High-force node incidence count mismatch")
    nodes = nodes.sort_values(node_keys, ignore_index=True)
    edges = edges.sort_values(["geometry", "sim_idx", "edge_u", "edge_v"], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, frame in (("nodes", nodes), ("edges", edges)):
            frame.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes(1, 3)
            worksheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            for column, name in enumerate(frame.columns):
                worksheet.set_column(column, column, min(30, max(14, len(name) + 2)))
    metadata = {
        "dataset": "final_load", "force_split": "force_split2", "graph_view": "core",
        "threshold": threshold,
        "threshold_rule": "normal_force >= 2 * pooled 0deg mean over particle-particle contacts only",
        "population": "All 20 simulations per geometry; particle nodes and particle-particle edges only",
        "node_label_rule": "Incident to at least one high-force particle-particle edge; high_force_degree counts only these edges",
        "stored_high_nodes_reclassified_without_high_force_core_edges": stored_only_count,
        "difference_from_existing_job5": "Existing Job 5 uses stored node labels including wall incidence; this workbook recomputes particle-only node labels",
        "property_rule": "Existing complete-core-graph values; no recomputation on force-selected subgraphs",
        "angle_rule": "angle_with_zz is the stored acute contact-normal angle to loading z, in degrees; defined for edges only",
        "endpoint_rule": "u_/v_ columns are endpoint node labels and properties, not edge centralities",
        "ids": "sim_idx and node IDs are zero-based",
        "rows": {"nodes": len(nodes), "edges": len(edges)},
        "group_counts": {scope: frame.groupby(["geometry", "group"]).size().rename("count").reset_index().to_dict("records")
                         for scope, frame in (("nodes", nodes), ("edges", edges))},
        "plotting": {
            "source": "AnalysisScripts/PostAnalysis/PeriodicRubyPipeline/merge_results.py:job5_complete_distribution_plots",
            "pooling": "Pool all finite node/contact observations within each geometry and force group",
            "normalization": "Separate density=True histogram for each group; area equals one",
            "bins": "60 equal-width bins shared across geometries and force groups; unit-width integer bins when range permits",
            "display": "Stair-step density curves; constant values drawn as vertical dashed lines; log y if positive densities span >100-fold",
            "inference": "Simulation summaries and paired high-minus-non-high contrasts, not individual particles as replicates",
        },
        "sources": node_files + edge_files,
        "threshold_source": str(reference_path.relative_to(PROJECT)),
        "threshold_source_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output}: {len(nodes):,} nodes; {len(edges):,} edges; threshold {threshold:.16g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()
