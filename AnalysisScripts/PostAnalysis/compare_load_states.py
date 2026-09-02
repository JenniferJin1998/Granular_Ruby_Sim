#!/usr/bin/env python
"""Compare final-load and jamming-state graph properties."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_RESULTS_DIR = SCRIPT_DIR.parent.parent / "AnalysisResults"
DEFAULT_FINAL_DIR = ANALYSIS_RESULTS_DIR / "FinalLoadState" / "FullGraph_2mean_ref_geom"
DEFAULT_JAMMING_DIR = ANALYSIS_RESULTS_DIR / "JammingState" / "FullGraph_final_load_threshold"
DEFAULT_OUT_DIR = ANALYSIS_RESULTS_DIR / "LoadStateComparison" / "Final_vs_Jamming_final_threshold"

NODE_KEYS = ["geometry", "sim_idx", "node_id"]
GRAPH_KEYS = ["geometry", "sim_idx"]
EXCLUDE_NUMERIC = {"node_id", "sim_idx"}


def numeric_feature_columns(df_a: pd.DataFrame, df_b: pd.DataFrame, keys: list[str]) -> list[str]:
    shared = [c for c in df_a.columns if c in df_b.columns and c not in set(keys) | EXCLUDE_NUMERIC]
    cols = []
    for col in shared:
        if pd.api.types.is_bool_dtype(df_a[col]) or pd.api.types.is_bool_dtype(df_b[col]):
            continue
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            cols.append(col)
    return cols


def add_change_columns(merged: pd.DataFrame, features: list[str], a_suffix: str, b_suffix: str) -> pd.DataFrame:
    out = merged.copy()
    for feature in features:
        a = f"{feature}_{a_suffix}"
        b = f"{feature}_{b_suffix}"
        out[f"{feature}_delta"] = out[a] - out[b]
        denom = out[b].replace(0, np.nan)
        out[f"{feature}_relative_delta"] = out[f"{feature}_delta"] / denom
    return out


def make_long_delta_table(wide: pd.DataFrame, keys: list[str], features: list[str]) -> pd.DataFrame:
    records = []
    for feature in features:
        cols = keys + [
            f"{feature}_jamming",
            f"{feature}_final",
            f"{feature}_delta",
            f"{feature}_relative_delta",
        ]
        tmp = wide[cols].copy()
        tmp.insert(len(keys), "feature", feature)
        tmp = tmp.rename(
            columns={
                f"{feature}_jamming": "jamming",
                f"{feature}_final": "final",
                f"{feature}_delta": "delta_final_minus_jamming",
                f"{feature}_relative_delta": "relative_delta",
            }
        )
        records.append(tmp)
    return pd.concat(records, ignore_index=True)


def compare_node_features(final_dir: Path, jamming_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_nodes = pd.read_csv(final_dir / "node_features.csv")
    jamming_nodes = pd.read_csv(jamming_dir / "node_features.csv")

    final_nodes = final_nodes[final_nodes["is_wall"] == False].copy()  # noqa: E712
    jamming_nodes = jamming_nodes[jamming_nodes["is_wall"] == False].copy()  # noqa: E712

    features = numeric_feature_columns(final_nodes, jamming_nodes, NODE_KEYS)
    merged = jamming_nodes[NODE_KEYS + features].merge(
        final_nodes[NODE_KEYS + features],
        on=NODE_KEYS,
        how="inner",
        suffixes=("_jamming", "_final"),
        validate="one_to_one",
    )
    wide = add_change_columns(merged, features, "final", "jamming")
    long = make_long_delta_table(wide, NODE_KEYS, features)

    wide.to_csv(out_dir / "node_property_changes_wide.csv", index=False)
    long.to_csv(out_dir / "node_property_changes_long.csv", index=False)
    return wide, long


def compare_graph_features(final_dir: Path, jamming_dir: Path, out_dir: Path) -> pd.DataFrame:
    final_graphs = pd.read_csv(final_dir / "graph_features.csv")
    jamming_graphs = pd.read_csv(jamming_dir / "graph_features.csv")

    features = numeric_feature_columns(final_graphs, jamming_graphs, GRAPH_KEYS)
    merged = jamming_graphs[GRAPH_KEYS + features].merge(
        final_graphs[GRAPH_KEYS + features],
        on=GRAPH_KEYS,
        how="inner",
        suffixes=("_jamming", "_final"),
        validate="one_to_one",
    )
    wide = add_change_columns(merged, features, "final", "jamming")
    wide.to_csv(out_dir / "graph_property_changes_wide.csv", index=False)
    return wide


def summarize_mean_changes(node_long: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    grouped = (
        node_long.groupby(["geometry", "feature"], dropna=False)
        .agg(
            n_nodes=("delta_final_minus_jamming", "size"),
            jamming_mean=("jamming", "mean"),
            final_mean=("final", "mean"),
            mean_delta_final_minus_jamming=("delta_final_minus_jamming", "mean"),
            median_delta_final_minus_jamming=("delta_final_minus_jamming", "median"),
            std_delta_final_minus_jamming=("delta_final_minus_jamming", "std"),
        )
        .reset_index()
    )
    grouped["relative_mean_delta"] = (
        grouped["mean_delta_final_minus_jamming"] / grouped["jamming_mean"].replace(0, np.nan)
    )

    pooled = (
        node_long.groupby(["feature"], dropna=False)
        .agg(
            n_nodes=("delta_final_minus_jamming", "size"),
            jamming_mean=("jamming", "mean"),
            final_mean=("final", "mean"),
            mean_delta_final_minus_jamming=("delta_final_minus_jamming", "mean"),
            median_delta_final_minus_jamming=("delta_final_minus_jamming", "median"),
            std_delta_final_minus_jamming=("delta_final_minus_jamming", "std"),
        )
        .reset_index()
    )
    pooled.insert(0, "geometry", "all")
    pooled["relative_mean_delta"] = (
        pooled["mean_delta_final_minus_jamming"] / pooled["jamming_mean"].replace(0, np.nan)
    )

    out = pd.concat([grouped, pooled], ignore_index=True)
    out.to_csv(out_dir / "node_mean_property_changes.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--jamming-dir", type=Path, default=DEFAULT_JAMMING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _, node_long = compare_node_features(args.final_dir.resolve(), args.jamming_dir.resolve(), out_dir)
    compare_graph_features(args.final_dir.resolve(), args.jamming_dir.resolve(), out_dir)
    summary = summarize_mean_changes(node_long, out_dir)

    print(f"Wrote comparison outputs to: {out_dir}")
    print(f"Node-change rows: {len(node_long)}")
    print(f"Mean-change rows: {len(summary)}")


if __name__ == "__main__":
    main()
