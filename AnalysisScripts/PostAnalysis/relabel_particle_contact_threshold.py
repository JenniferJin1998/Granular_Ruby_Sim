#!/usr/bin/env python3
"""Relabel graphs using 2 x mean 0deg particle-particle normal force."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from relabel_jamming_with_final_threshold import (
    copy_context_files,
    load_pickle,
    relabel_graph_dict,
    save_pickle,
    write_feature_tables,
)


def particle_contact_threshold(graph_dict: dict) -> tuple[float, dict]:
    forces = np.asarray([
        data["normal_force"]
        for graph in graph_dict["0deg"]["core"]
        for _, _, data in graph.edges(data=True)
        if "normal_force" in data
    ], dtype=float)
    forces = forces[np.isfinite(forces)]
    if not len(forces):
        raise ValueError("No finite particle-particle normal forces found for 0deg")
    mean_force = float(np.mean(forces))
    threshold = float(2.0 * mean_force)
    return threshold, {
        "mode": "reference[0deg]_particle_particle_contacts_only",
        "thresholds": {"0deg": threshold},
        "reference_mean_normal_force": mean_force,
        "reference_contact_count": int(len(forces)),
        "comparison_operator": ">=",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_dict = load_pickle(input_dir / "graph_dict_labeled.pkl")
    threshold, threshold_info = particle_contact_threshold(graph_dict)
    high_force_records, _ = relabel_graph_dict(
        graph_dict, threshold, f"{input_dir}/graph_dict_labeled.pkl: 0deg core edges"
    )
    threshold_info["source"] = f"{input_dir}/graph_dict_labeled.pkl: pooled 0deg core edges"

    save_pickle(graph_dict, output_dir / "graph_dict_labeled.pkl")
    save_pickle(threshold_info, output_dir / "high_force_threshold_info.pkl")
    pd.DataFrame(high_force_records, columns=["geometry", "sim_idx", "node1", "node2"]).to_csv(
        output_dir / "high_force_edges.csv", index=False
    )
    write_feature_tables(graph_dict, output_dir)
    copy_context_files(input_dir, output_dir)
    for name in ("periodic_geometry_metadata.json", "geometry_estimate.json"):
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    counts = {}
    for geometry, views in graph_dict.items():
        particle_high = sum(
            int(data.get("is_high_force", False))
            for graph in views["core"] for _, _, data in graph.edges(data=True)
        )
        full_high = sum(
            int(data.get("is_high_force", False))
            for graph in views["full"] for _, _, data in graph.edges(data=True)
        )
        counts[geometry] = {"particle_particle_high_contacts": particle_high, "all_high_contacts": full_high}
    (output_dir / "threshold_audit.json").write_text(json.dumps({**threshold_info, "counts_by_geometry": counts}, indent=2) + "\n")
    (output_dir / "README.md").write_text(f"""# Force split 2 graph data

The graph topology and properties come from `{input_dir}`. Only stored
high-force edge/node labels were regenerated.

```text
threshold = 2 * mean(normal_force over pooled 0deg particle-particle contacts)
          = {threshold:.17g}
comparison operator = >=
```

Wall contacts are excluded from the reference mean. The resulting threshold is
applied to all geometries. Wall contacts are still labeled by that threshold in
the full graph, matching the original tagging semantics; connected force
clusters use particle-particle edges only.
""")
    print(json.dumps({"output": str(output_dir), "threshold": threshold, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
