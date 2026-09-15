#!/usr/bin/env python3
"""Plot signed boundary-property differences relative to the 0-degree geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATASET_ROOTS = {
    name: PROJECT / "AnalysisResults" / name / "1_network_property_comparison" / "boundary_layers"
    for name in ("final_load", "jamming")
}
DATASET_ROOTS.update({
    "periodic_force_split2": PROJECT / "AnalysisResults" / "periodic_boundaries" / "2026-08-03" / "1_network_property_comparison" / "force_splits" / "force_split2" / "boundary_layers",
    "final_load_force_split2": PROJECT / "AnalysisResults" / "final_load" / "1_network_property_comparison" / "force_splits" / "force_split2" / "boundary_layers",
    "jamming_final_threshold": PROJECT / "AnalysisResults" / "jamming" / "1_network_property_comparison" / "boundary_layers",
})
REFERENCES = ("all", "top_bottom", "top", "bottom")
TARGET_GEOMETRIES = ("15deg", "30deg", "45deg")
COLORS = {"15deg": "#2CA02C", "30deg": "#FF7F0E", "45deg": "#9467BD"}
SHELL_ORDER = ("0 (boundary)", "1", "rest (>=2/unreached)")


def signed_differences(summary: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Subtract pooled 0-degree mean/median from each requested geometry."""
    reference = summary[summary.geometry == "0deg"].copy()
    rows = []
    for target_geometry in [geometry for geometry in TARGET_GEOMETRIES if geometry in set(summary.geometry)]:
        target = summary[summary.geometry == target_geometry].copy()
        merged = target.merge(reference, on=group_columns, suffixes=("_target", "_0deg"), validate="one_to_one")
        for row in merged.itertuples(index=False):
            record = {column: getattr(row, column) for column in group_columns}
            record.update({
                "comparison": f"{target_geometry}_minus_0deg",
                "target_geometry": target_geometry,
                "reference_geometry": "0deg",
                "n_target": int(row.n_target),
                "n_0deg": int(row.n_0deg),
                "mean_target": float(row.mean_target),
                "mean_0deg": float(row.mean_0deg),
                "mean_difference_target_minus_0deg": float(row.mean_target - row.mean_0deg),
                "median_target": float(row.median_target),
                "median_0deg": float(row.median_0deg),
                "median_difference_target_minus_0deg": float(row.median_target - row.median_0deg),
            })
            rows.append(record)
    return pd.DataFrame(rows)


def plot_property_differences(contrasts: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for prop, data in contrasts.groupby("property", sort=True):
        target_geometries = [geometry for geometry in TARGET_GEOMETRIES if geometry in set(data.target_geometry)]
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, constrained_layout=True)
        for ax, statistic in zip(axes, ("mean", "median")):
            value_column = f"{statistic}_difference_target_minus_0deg"
            for geometry in target_geometries:
                group = data[data.target_geometry == geometry].set_index("boundary_shell").reindex(SHELL_ORDER)
                ax.plot(range(len(SHELL_ORDER)), group[value_column], marker="o", linewidth=1.5,
                        color=COLORS[geometry], label=f"{geometry.replace('deg', '°')} − 0°")
            ax.axhline(0, color="black", linewidth=.8)
            ax.set(ylabel=f"pooled {statistic} difference", title=f"{statistic.title()}: target geometry − 0°")
            ax.grid(axis="y", alpha=.20)
        axes[-1].set_xticks(range(len(SHELL_ORDER)), SHELL_ORDER)
        axes[0].legend(frameon=False, ncol=3, fontsize=9)
        fig.suptitle(f"{data.entity.iloc[0].title()} {prop}: geometry differences by boundary layer", fontsize=13)
        fig.savefig(output / f"{prop}.png", dpi=180)
        plt.close(fig)


def boundary_group_order(view: str) -> list[str]:
    if view == "0_vs_rest":
        return ["distance 0", "rest (>=1/unreachable)"]
    return ["distance 0", "distance 1", "rest (>=2/unreachable)"]


def plot_force_differences(contrasts: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for (prop, view), data in contrasts.groupby(["property", "boundary_view"], sort=True):
        groups = boundary_group_order(view)
        forces = ("force cluster", "non-force")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, constrained_layout=True)
        for row_index, statistic in enumerate(("mean", "median")):
            value_column = f"{statistic}_difference_target_minus_0deg"
            for column_index, force_group in enumerate(forces):
                ax = axes[row_index, column_index]
                subset = data[data.force_group == force_group]
                for geometry in [item for item in TARGET_GEOMETRIES if item in set(subset.target_geometry)]:
                    line = subset[subset.target_geometry == geometry].set_index("boundary_group").reindex(groups)
                    ax.plot(range(len(groups)), line[value_column], marker="o", linewidth=1.5,
                            color=COLORS[geometry], label=f"{geometry.replace('deg', '°')} − 0°")
                ax.axhline(0, color="black", linewidth=.8)
                ax.set(title=force_group, ylabel=f"pooled {statistic} difference" if column_index == 0 else "")
                ax.grid(axis="y", alpha=.20)
        for ax in axes[-1]:
            ax.set_xticks(range(len(groups)), groups, rotation=15, ha="right")
        axes[0, 0].legend(frameon=False, ncol=3, fontsize=9)
        fig.suptitle(f"{data.entity.iloc[0].title()} {prop}: force split, {view}; target geometry − 0°", fontsize=13)
        fig.savefig(output / f"{prop}_{view}.png", dpi=180)
        plt.close(fig)


def analyze(dataset: str, reference: str) -> None:
    boundary = DATASET_ROOTS[dataset] / f"{reference}_surfaces"
    tables = boundary / "tables"
    output = boundary / "geometry_minus_0"
    output_tables = output / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)
    for level in ("node", "edge"):
        summary = pd.read_csv(tables / f"{level}_property_summary_by_shell.csv")
        contrasts = signed_differences(summary, ["entity", "property", "boundary_shell"])
        contrasts.to_csv(output_tables / f"{level}_property_geometry_minus_0.csv", index=False)
        plot_property_differences(contrasts, output / "property" / level)

        force_summary = pd.read_csv(tables / f"{level}_force_split_pooled_summary.csv")
        force_contrasts = signed_differences(
            force_summary, ["entity", "property", "boundary_view", "boundary_group", "force_group"]
        )
        force_contrasts.to_csv(output_tables / f"{level}_force_split_geometry_minus_0.csv", index=False)
        plot_force_differences(force_contrasts, output / "force_split" / level)

    readme = f"""# Geometry differences relative to 0 degrees

This folder contains signed pooled-property contrasts for `{dataset}` with the
`{reference}` boundary reference. Every value is target geometry minus 0°:
`15°−0°`, `30°−0°`, and `45°−0°`.

- `property/{{node,edge}}/`: mean and median differences at distance 0, distance
  1, and rest.
- `force_split/{{node,edge}}/`: the same differences shown separately for force
  cluster and non-force populations.
- `tables/`: exact pooled means, medians, sample counts, and signed differences.

Positive means the target geometry has the larger property value; negative
means 0° has the larger value. These are pooled descriptive effects. Use the
parent `tables/*_simulation_angle_tests.csv` and
`tables/*_force_split_simulation_tests.csv` for simulation-replicate inference.
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_ROOTS, required=True)
    parser.add_argument("--reference", choices=REFERENCES, required=True)
    args = parser.parse_args()
    analyze(args.dataset, args.reference)


if __name__ == "__main__":
    main()
