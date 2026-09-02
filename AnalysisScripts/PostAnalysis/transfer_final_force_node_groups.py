#!/usr/bin/env python
"""Group jamming-state node properties by final-state high-force node labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_RESULTS_DIR = SCRIPT_DIR.parent.parent / "AnalysisResults"
DEFAULT_FINAL_DIR = ANALYSIS_RESULTS_DIR / "FinalLoadState" / "FullGraph_2mean_ref_geom"
DEFAULT_JAMMING_DIR = ANALYSIS_RESULTS_DIR / "JammingState" / "FullGraph_final_load_threshold"
DEFAULT_OUT_DIR = ANALYSIS_RESULTS_DIR / "JammingState" / "FinalForceNodeGroups"

NODE_KEYS = ["geometry", "sim_idx", "node_id"]
EXCLUDE_NUMERIC = {"sim_idx", "node_id"}


def numeric_node_features(df: pd.DataFrame) -> list[str]:
    features = []
    for col in df.columns:
        if col in set(NODE_KEYS) | EXCLUDE_NUMERIC:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def summarize_by_group(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records = []
    group_specs = [
        ("final_high_force", df["final_is_force_chain_node"] == True),  # noqa: E712
        ("final_non_high_force", df["final_is_force_chain_node"] == False),  # noqa: E712
        ("all", pd.Series(True, index=df.index)),
    ]

    for geometry, geom_df in df.groupby("geometry", sort=False):
        for group_name, mask in group_specs:
            sub = geom_df[mask.loc[geom_df.index]]
            for feature in features:
                vals = pd.to_numeric(sub[feature], errors="coerce").dropna()
                records.append(
                    {
                        "geometry": geometry,
                        "feature": feature,
                        "group": group_name,
                        "mean": vals.mean(),
                        "stderr": vals.std(ddof=1) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                        "feature_type": "node",
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--jamming-dir", type=Path, default=DEFAULT_JAMMING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    final_nodes = pd.read_csv(args.final_dir / "node_features.csv")
    jamming_nodes = pd.read_csv(args.jamming_dir / "node_features.csv")

    final_labels = final_nodes[final_nodes["is_wall"] == False][  # noqa: E712
        NODE_KEYS + ["high_force_degree", "is_force_chain_node", "force_chain_role"]
    ].rename(
        columns={
            "high_force_degree": "final_high_force_degree",
            "is_force_chain_node": "final_is_force_chain_node",
            "force_chain_role": "final_force_chain_role",
        }
    )

    jamming_particle_nodes = jamming_nodes[jamming_nodes["is_wall"] == False].copy()  # noqa: E712
    merged = jamming_particle_nodes.merge(
        final_labels,
        on=NODE_KEYS,
        how="left",
        validate="one_to_one",
    )

    missing = merged["final_is_force_chain_node"].isna().sum()
    if missing:
        raise ValueError(f"{missing} jamming particle nodes did not match final-state node labels")

    features = numeric_node_features(jamming_particle_nodes)
    stats = summarize_by_group(merged, features)

    merged.to_csv(out_dir / "jamming_node_features_with_final_force_labels.csv", index=False)
    stats.to_csv(out_dir / "jamming_node_stats_by_final_force_group.csv", index=False)

    readme = f"""# Jamming node properties grouped by final-state high-force labels

Inputs:

- Final labels: `{args.final_dir / "node_features.csv"}`
- Jamming properties: `{args.jamming_dir / "node_features.csv"}`

Nodes are matched by `geometry`, `sim_idx`, and `node_id`.

Outputs:

- `jamming_node_features_with_final_force_labels.csv`: jamming node properties plus `final_*` high-force labels.
- `jamming_node_stats_by_final_force_group.csv`: mean/stderr by geometry for `final_high_force`, `final_non_high_force`, and `all`.

Edge grouping is intentionally not transferred because contacts/edges change between load states and do not have a stable edge id here.
"""
    (out_dir / "README.md").write_text(readme)

    print(f"Wrote transferred-label node table and stats to: {out_dir}")
    print(f"Matched particle nodes: {len(merged)}")
    print(f"Features summarized: {len(features)}")


if __name__ == "__main__":
    main()
