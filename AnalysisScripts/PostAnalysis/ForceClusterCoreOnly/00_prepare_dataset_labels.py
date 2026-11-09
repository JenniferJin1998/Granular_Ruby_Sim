#!/usr/bin/env python3
"""Write particles.csv and contacts.csv for the datasets other than final_load_force.

Run with BC_DATASET set (see core_only_common.py). Output columns match the final-load
export: angle, sim_idx, node_id, group, degree, closeness, betweenness, stress_hydro, x, y, z
(z = loading axis) and contacts angle, sim_idx, edge_u, edge_v, normal_force, angle_with_zz.

  final_load_stress: final-load particles and contacts (00_workbook_labels); group =
      high_force when stress_hydro <= system mean - 1 SD (SD with ddof=1); the force-cluster
      label is kept as force_cluster (True/False).
  periodic_force:    force_split2 node_features / edge_features (core edges only); group =
      high_force when the particle has a particle-particle contact with normal force >= the
      stored threshold (2 x pooled 0deg mean of particle-particle contacts).
  experiment_stress: Results/14 graphs (LoadStep2, t5 contacts, core edges); one sample per
      angle (sim_idx 0); group = high_force when stress_hydro <= sample mean - 1 SD; particles
      without stress (no FF grain) are 'unlabeled' (kept for the graph, left out of statistics).
      The experiment loading axis is y: saved x, y, z = original x, z, y (originals kept as
      x_orig, y_orig, z_orig), so z is the loading axis as in the simulations.
More negative hydrostatic stress = more compressive.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core_only_common import (CONTACTS_CSV, DATASET_NAME, EXPERIMENT_RESULTS, FINAL_ROOT, LABELS_DIR,
                              LABELS_METADATA, PARTICLES_CSV, PERIODIC_CASE, UNLABELED, sha256)

PERIODIC_SPLIT = PERIODIC_CASE / "0_graph_and_basic_stats" / "force_splits" / "force_split2"
EXPERIMENT_GRAPHS = EXPERIMENT_RESULTS / "14_ExperimentalGraphNetwork"
EXPERIMENT_SAMPLES = {"Config0_Sam1": "0deg", "Config2_Sam2": "30deg"}


def stress_groups(particles, keys=("angle", "sim_idx")):
    """high_force (= high-stress) when stress_hydro <= mean - 1 SD of the particle's system."""
    by = particles.groupby(list(keys)).stress_hydro
    cut = by.transform("mean") - by.transform("std")
    group = np.where(particles.stress_hydro <= cut, "high_force", "non_high_force")
    return pd.Series(np.where(particles.stress_hydro.isna(), UNLABELED, group), index=particles.index), cut


def check_degree(particles, contacts):
    ends = pd.concat([contacts[["angle", "sim_idx", "edge_u"]].rename(columns={"edge_u": "node_id"}),
                      contacts[["angle", "sim_idx", "edge_v"]].rename(columns={"edge_v": "node_id"})])
    degree = ends.groupby(["angle", "sim_idx", "node_id"]).size()
    stored = particles.set_index(["angle", "sim_idx", "node_id"]).degree
    if not stored.eq(degree.reindex(stored.index).fillna(0).astype(int)).all():
        raise ValueError("Contact list does not reproduce the stored degrees")


def final_load_stress():
    source = FINAL_ROOT / "00_workbook_labels"
    particles = pd.read_csv(source / "particles.csv")
    contacts = pd.read_csv(source / "contacts.csv")
    # Original force-cluster label, named so the high_force -> high_stress relabeling leaves it alone.
    particles["force_cluster"] = particles.pop("group").eq("high_force")
    particles = particles.drop(columns=["high_force_degree", "is_force_chain_node"], errors="ignore")
    particles["group"], particles["stress_threshold"] = stress_groups(particles)
    info = {"sources": [str(source / "particles.csv"), str(source / "contacts.csv")],
            "label_rule": "high-stress = stress_hydro <= mean - 1 SD (ddof=1) of the particle's system; "
                          "more negative = more compressive",
            "force_cluster": "original force-cluster label (True/False) kept for reference"}
    return particles, contacts, info


def periodic_force():
    nodes = pd.read_csv(PERIODIC_SPLIT / "node_features.csv",
                        usecols=["geometry", "sim_idx", "node_id", "is_wall", "x", "y", "z", "stress_hydro",
                                 "degree", "closeness", "betweenness"])
    nodes = nodes[~nodes.is_wall.astype(bool)].drop(columns="is_wall").rename(columns={"geometry": "angle"})
    edges = pd.read_csv(PERIODIC_SPLIT / "edge_features.csv",
                        usecols=["geometry", "sim_idx", "node1", "node2", "is_core_edge", "normal_force",
                                 "angle_with_zz", "is_high_force"])
    edges = edges[edges.is_core_edge.astype(bool)].rename(columns={"geometry": "angle"})
    u, v = edges.node1.astype(int), edges.node2.astype(int)
    contacts = pd.DataFrame({"angle": edges.angle, "sim_idx": edges.sim_idx, "edge_u": np.minimum(u, v),
                             "edge_v": np.maximum(u, v), "is_high_force": edges.is_high_force.astype(bool),
                             "normal_force": edges.normal_force, "angle_with_zz": edges.angle_with_zz})
    audit = json.loads((PERIODIC_SPLIT / "threshold_audit.json").read_text())
    threshold = audit["thresholds"]["0deg"]
    if not ((contacts.normal_force >= threshold) == contacts.is_high_force).all():
        raise ValueError("Stored is_high_force differs from normal_force >= threshold")
    high = contacts[contacts.is_high_force]
    ends = pd.concat([high[["angle", "sim_idx", "edge_u"]].rename(columns={"edge_u": "node_id"}),
                      high[["angle", "sim_idx", "edge_v"]].rename(columns={"edge_v": "node_id"})])
    high_degree = ends.groupby(["angle", "sim_idx", "node_id"]).size()
    key = pd.MultiIndex.from_frame(nodes[["angle", "sim_idx", "node_id"]])
    nodes["high_force_degree"] = high_degree.reindex(key).fillna(0).astype(int).to_numpy()
    nodes["group"] = np.where(nodes.high_force_degree > 0, "high_force", "non_high_force")
    info = {"sources": [str(PERIODIC_SPLIT / "node_features.csv"), str(PERIODIC_SPLIT / "edge_features.csv")],
            "threshold": threshold, "threshold_rule": audit["mode"] + ", normal_force >= threshold",
            "label_rule": "high-force = at least one particle-particle contact with normal force >= threshold",
            "geometry": "x and y periodic, z (loading) nonperiodic; no radial analysis",
            "betweenness": "core graph (particle-particle contacts, no walls)"}
    return nodes, contacts, info


def experiment_stress():
    particles, contacts = [], []
    for sample, angle in EXPERIMENT_SAMPLES.items():
        nodes = pd.read_csv(EXPERIMENT_GRAPHS / sample / "node_features.csv")
        nodes = nodes[~nodes.is_wall.astype(bool)]
        frame = pd.DataFrame({"angle": angle, "sim_idx": 0, "sample": sample, "node_id": nodes.node_id,
                              "degree": nodes.degree, "closeness": nodes.closeness,
                              "betweenness": nodes.betweenness, "stress_hydro": nodes.stress_hydro,
                              "x": nodes.x, "y": nodes.z, "z": nodes.y,
                              "x_orig": nodes.x, "y_orig": nodes.y, "z_orig": nodes.z})
        particles.append(frame)
        edges = pd.read_csv(EXPERIMENT_GRAPHS / sample / "edge_features.csv")
        edges = edges[edges.is_core_edge.astype(bool)]
        u, v = edges.node1.astype(int), edges.node2.astype(int)
        contacts.append(pd.DataFrame({"angle": angle, "sim_idx": 0, "edge_u": np.minimum(u, v),
                                      "edge_v": np.maximum(u, v), "normal_force": edges.normal_force,
                                      "angle_with_zz": edges.angle_with_zz}))
    particles = pd.concat(particles, ignore_index=True)
    contacts = pd.concat(contacts, ignore_index=True)
    particles["group"], particles["stress_threshold"] = stress_groups(particles)
    info = {"sources": [str(EXPERIMENT_GRAPHS / s) for s in EXPERIMENT_SAMPLES],
            "samples": EXPERIMENT_SAMPLES,
            "label_rule": "high-stress = stress_hydro <= mean - 1 SD (ddof=1) of the sample; more negative = "
                          "more compressive; particles without stress are 'unlabeled'",
            "units": "positions mm, stress GPa",
            "axes": "saved x, y, z = original x, z, y; z (original y) is the loading axis",
            "replicates": "one sample per angle: no confidence intervals over systems"}
    return particles, contacts, info


def main():
    builders = {"final_load_stress": final_load_stress, "periodic_force": periodic_force,
                "experiment_stress": experiment_stress}
    if DATASET_NAME not in builders:
        raise ValueError(f"Set BC_DATASET to one of {sorted(builders)} (final_load_force uses "
                         "00_export_workbook_labels.py)")
    particles, contacts, info = builders[DATASET_NAME]()
    check_degree(particles, contacts)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    particles.to_csv(PARTICLES_CSV, index=False)
    contacts.to_csv(CONTACTS_CSV, index=False)
    counts = particles.groupby(["angle", "group"]).size().unstack(fill_value=0)
    per_system = particles[particles.group != UNLABELED].groupby(["angle", "sim_idx"]).group.apply(
        lambda g: (g == "high_force").mean())
    info.update({"dataset": DATASET_NAME, "particles": len(particles), "contacts": len(contacts),
                 "group_counts": {a: {k: int(n) for k, n in counts.loc[a].items()} for a in counts.index},
                 "high_share_per_system": {a: [round(per_system.loc[a].min(), 4), round(per_system.loc[a].max(), 4)]
                                           for a in per_system.index.get_level_values(0).unique()},
                 "source_sha256": {s: sha256(s) for s in info["sources"] if s.endswith(".csv")},
                 "walls": "treated as unknown; no wall nodes or wall contacts are used",
                 "group_names": "tables of stress datasets say high_stress / non_high_stress"})
    LABELS_METADATA.write_text(json.dumps(info, indent=2, default=str))
    print(counts.to_string())
    print("high share per system (min, max):", info["high_share_per_system"])
    print(f"Saved {PARTICLES_CSV} and {CONTACTS_CSV}")


if __name__ == "__main__":
    main()
