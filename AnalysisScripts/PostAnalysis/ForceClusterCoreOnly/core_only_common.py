"""Shared paths, labels, and statistics for the core-only betweenness analysis.

The dataset is chosen with the environment variable BC_DATASET (default final_load_force):

  final_load_force   final load, labels of the collaborator workbook
                     (final_load_force_split2_populations.xlsx, exported by
                     00_export_workbook_labels.py): high-force = at least one high-force
                     particle-particle contact (member of a force cluster).
  final_load_stress  final load, same particles and contacts; high-stress = hydrostatic stress
                     at or below (more compressive than) its system's mean - 1 SD.
  periodic_force     periodic boundaries (2026-08-03), force_split2: high-force = at least one
                     particle-particle contact with normal force >= the stored threshold.
  experiment_stress  experiment LoadStep2 graphs (Results/14), one sample per angle;
                     high-stress = hydrostatic stress <= that sample's mean - 1 SD.

Walls are treated as unknown everywhere: only particles and particle-particle contacts
are used and betweenness is the core-graph (no-wall) value. Inputs for the last three are
written by 00_prepare_dataset_labels.py.

Internally the two groups are always called "high_force" / "non_high_force". For the stress
datasets every saved figure and table is relabeled to "high-stress" (see _install_relabeling).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yfjin/matplotlib")

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
CASE = PROJECT / "AnalysisResults" / "final_load"
WORKBOOK = (CASE / "2_force_cluster_comparison" / "force_split2"
            / "final_load_force_split2_populations.xlsx")
NODE_FEATURES = CASE / "0_graph_and_basic_stats" / "force_splits" / "force_split2" / "node_features.csv"
FINAL_ROOT = CASE / "04_force_cluster_analysis_core_only"
PERIODIC_CASE = PROJECT / "AnalysisResults" / "periodic_boundaries" / "2026-08-03"
EXPERIMENT_RESULTS = PROJECT.parent / "Granular_RubyExp_202608" / "Results"

DATASETS = {
    "final_load_force": {
        "output_root": FINAL_ROOT, "labels_dir": "00_workbook_labels", "kind": "force",
        "title": "final load, force_split2, core only", "y_scale": 1e5,
        "y_label": r"Hydrostatic stress ($\times 10^{-5}$)", "location": "sim_zeff", "radial": True,
    },
    "final_load_stress": {
        "output_root": FINAL_ROOT / "10_high_stress_classification", "labels_dir": "00_labels",
        "kind": "stress", "title": "final load, high-stress = mean - 1 SD per system, core only",
        "y_scale": 1e5, "y_label": r"Hydrostatic stress ($\times 10^{-5}$)", "location": "sim_zeff",
        "radial": True,
    },
    "periodic_force": {
        "output_root": PERIODIC_CASE / "5_betweenness_force_cluster_core_only", "labels_dir": "00_labels",
        "kind": "force", "title": "periodic boundaries, force_split2, core only", "y_scale": 1e10,
        "y_label": r"Hydrostatic stress ($\times 10^{-10}$)", "location": "z", "radial": False,
    },
    "experiment_stress": {
        "output_root": EXPERIMENT_RESULTS / "18_BetweennessHighStress", "labels_dir": "00_labels",
        "kind": "stress", "title": "experiment LS2, high-stress = mean - 1 SD per sample, core only",
        "y_scale": 1e3, "y_label": "Hydrostatic stress (MPa)", "location": "r_z", "radial": True,
    },
}
DATASET_NAME = os.environ.get("BC_DATASET", "final_load_force")
if DATASET_NAME not in DATASETS:
    raise ValueError(f"BC_DATASET must be one of {sorted(DATASETS)}, got {DATASET_NAME!r}")
DATASET = DATASETS[DATASET_NAME]
IS_STRESS = DATASET["kind"] == "stress"

OUTPUT_ROOT = DATASET["output_root"]
LABELS_DIR = OUTPUT_ROOT / DATASET["labels_dir"]
PARTICLES_CSV = LABELS_DIR / "particles.csv"
CONTACTS_CSV = LABELS_DIR / "contacts.csv"
LABELS_METADATA = LABELS_DIR / "provenance.json"

GROUPS = ("high_force", "non_high_force")
GROUP_LABELS = ({"high_force": "High-stress (mean - 1 SD)", "non_high_force": "Non-high-stress"} if IS_STRESS
                else {"high_force": "High-force (force-cluster)", "non_high_force": "Non-high-force"})
GROUP_COLORS = {"high_force": "#D95F02", "non_high_force": "#2474B5"}
UNLABELED = "unlabeled"

# Display and table text for the stress datasets (longest patterns first).
_TEXT_SWAPS = (("High-force (force-cluster)", "High-stress (mean - 1 SD)"),
               ("Non-high-force", "Non-high-stress"), ("non-high-force", "non-high-stress"),
               ("High-force", "High-stress"), ("high-force", "high-stress"),
               ("force-cluster", "high-stress"), ("Force-cluster", "High-stress"),
               ("force cluster", "high-stress group"))
_COLUMN_SWAPS = (("non_high_force", "non_high_stress"), ("high_force", "high_stress"))


def relabel_text(text):
    if not IS_STRESS or not isinstance(text, str):
        return text
    for old, new in _TEXT_SWAPS:
        text = text.replace(old, new)
    return text


def _relabel_name(name):
    if not isinstance(name, str):
        return name
    for old, new in _COLUMN_SWAPS:
        name = name.replace(old, new)
    return name


def _install_relabeling():
    """Stress datasets: rename high_force -> high_stress in every saved table and figure."""
    import matplotlib.figure
    import matplotlib.text

    original_savefig = matplotlib.figure.Figure.savefig
    original_to_csv = pd.DataFrame.to_csv

    def savefig(fig, *args, **kwargs):
        for text in fig.findobj(matplotlib.text.Text):
            new = relabel_text(text.get_text())
            if new != text.get_text():
                text.set_text(new)
        return original_savefig(fig, *args, **kwargs)

    def to_csv(frame, *args, **kwargs):
        out = frame.rename(columns=_relabel_name)
        for column in out.columns:
            if out[column].dtype == object:
                out[column] = out[column].map(_relabel_name)
        return original_to_csv(out, *args, **kwargs)

    matplotlib.figure.Figure.savefig = savefig
    pd.DataFrame.to_csv = to_csv


if IS_STRESS:
    _install_relabeling()


def angle_key(label):
    return float(re.search(r"(\d+(?:\.\d+)?)", str(label)).group(1))


def sorted_angles(values):
    return sorted(pd.unique(values), key=angle_key)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def force_threshold():
    """High-force contact threshold of the force datasets (normal force >= threshold)."""
    if DATASET_NAME == "final_load_force":
        return json.loads(WORKBOOK.with_suffix(".json").read_text())["threshold"]
    return json.loads(LABELS_METADATA.read_text())["threshold"]


def load_particles(include_unlabeled=False):
    """Particles with internal group names. Unlabeled particles (experiment: no stress data) are
    dropped unless include_unlabeled; their betweenness still comes from the full core graph."""
    if not PARTICLES_CSV.exists():
        raise FileNotFoundError(f"Prepare the labels first ({DATASET_NAME}): {PARTICLES_CSV}")
    particles = pd.read_csv(PARTICLES_CSV)
    particles["group"] = particles.group.replace({"high_stress": "high_force",
                                                  "non_high_stress": "non_high_force"})
    if not include_unlabeled:
        particles = particles[particles.group.isin(GROUPS)].reset_index(drop=True)
    return particles


def load_contacts():
    if not CONTACTS_CSV.exists():
        raise FileNotFoundError(f"Prepare the labels first ({DATASET_NAME}): {CONTACTS_CSV}")
    return pd.read_csv(CONTACTS_CSV)


def mean_ci(values):
    """Mean and 95% t interval over systems (the replicates); a single system gives its value
    and no interval."""
    from scipy import stats

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], np.nan, np.nan
    mean = values.mean()
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def summarize(frame, by, value_columns):
    rows = []
    for keys, sub in frame.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["systems"] = sub.sim_idx.nunique()
        for column in value_columns:
            row[f"{column}_mean"], row[f"{column}_ci_low"], row[f"{column}_ci_high"] = mean_ci(sub[column])
        rows.append(row)
    return pd.DataFrame(rows)


def angle_colors(angles):
    import matplotlib.pyplot as plt

    return dict(zip(angles, plt.cm.viridis(np.linspace(0.05, 0.85, len(angles)))))
