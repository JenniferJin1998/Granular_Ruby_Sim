#!/usr/bin/env python3
"""Export the collaborator workbook's particle labels and contacts to CSV.

Run with a Python that has openpyxl (the base Anaconda install; the
graph_analysis environment lacks it). Later steps read only these CSVs.

particles.csv: workbook nodes sheet (labels, betweenness) joined with
  stress_hydro and particle center x, y, z (m) from the workbook's own source,
  force_split2 node_features.csv
  (particle rows only). Betweenness is checked to be identical in both files.
contacts.csv:  workbook edges sheet (particle-particle contacts only).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core_only_common import (CONTACTS_CSV, LABELS_DIR, NODE_FEATURES, PARTICLES_CSV, WORKBOOK,
                              sha256)

KEYS = ["geometry", "sim_idx", "node_id"]


def main():
    nodes = pd.read_excel(WORKBOOK, sheet_name="nodes")
    edges = pd.read_excel(WORKBOOK, sheet_name="edges")
    if not ((nodes.group == "high_force") == (nodes.high_force_degree > 0)).all():
        raise ValueError("Workbook group does not match high_force_degree > 0")

    features = pd.read_csv(NODE_FEATURES, usecols=KEYS + ["is_wall", "betweenness", "stress_hydro",
                                                          "x", "y", "z"])
    features = features[~features.is_wall.astype(bool)].drop(columns="is_wall")
    particles = nodes.merge(features, on=KEYS, how="left", validate="one_to_one",
                            suffixes=("", "_node_features"))
    if particles.stress_hydro.isna().any():
        raise ValueError("Particles without stress_hydro in node_features.csv")
    if not np.allclose(particles.betweenness, particles.betweenness_node_features, rtol=1e-9, atol=0):
        raise ValueError("Workbook betweenness differs from node_features.csv")
    particles = particles.drop(columns="betweenness_node_features").rename(columns={"geometry": "angle"})

    contacts = edges[["geometry", "sim_idx", "edge_u", "edge_v", "is_high_force", "normal_force",
                      "angle_with_zz"]].rename(columns={"geometry": "angle"})
    # Degree from the contact list must reproduce the workbook degree.
    ends = pd.concat([contacts[["angle", "sim_idx", "edge_u"]].rename(columns={"edge_u": "node_id"}),
                      contacts[["angle", "sim_idx", "edge_v"]].rename(columns={"edge_v": "node_id"})])
    degree = ends.groupby(["angle", "sim_idx", "node_id"]).size()
    check = particles.set_index(["angle", "sim_idx", "node_id"]).degree
    if not check.eq(degree.reindex(check.index).fillna(0).astype(int)).all():
        raise ValueError("Contact list does not reproduce workbook degrees")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    particles.to_csv(PARTICLES_CSV, index=False)
    contacts.to_csv(CONTACTS_CSV, index=False)
    counts = particles.groupby(["angle", "group"]).size().unstack()
    (LABELS_DIR / "provenance.json").write_text(json.dumps({
        "workbook": str(WORKBOOK), "workbook_sha256": sha256(WORKBOOK),
        "node_features": str(NODE_FEATURES), "node_features_sha256": sha256(NODE_FEATURES),
        "label_rule": "high_force = at least one high-force particle-particle contact (workbook group)",
        "walls": "treated as unknown; no wall nodes or wall contacts are used",
        "stress_hydro_note": "particle stress from the simulation; physically includes wall loads",
        "checks": ["group == high_force_degree > 0", "betweenness identical to node_features.csv",
                   "contact list reproduces workbook degree"],
        "particles": len(particles), "contacts": len(contacts),
        "group_counts": {a: counts.loc[a].to_dict() for a in counts.index},
    }, indent=2))
    print(counts.to_string())
    print(f"Saved {PARTICLES_CSV} and {CONTACTS_CSV}")


if __name__ == "__main__":
    main()
