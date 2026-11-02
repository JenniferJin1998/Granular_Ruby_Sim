"""Shared paths, labels, and statistics for the core-only force-cluster analysis.

Every step reads the particle labels of the collaborator workbook
(final_load_force_split2_populations.xlsx), exported once by
00_export_workbook_labels.py. Walls are treated as unknown: only particles and
particle-particle contacts are used, and a particle is high-force when it has at
least one high-force particle-particle contact (it belongs to a force cluster).
"""

from __future__ import annotations

import hashlib
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
OUTPUT_ROOT = CASE / "04_force_cluster_analysis_core_only"
LABELS_DIR = OUTPUT_ROOT / "00_workbook_labels"
PARTICLES_CSV = LABELS_DIR / "particles.csv"
CONTACTS_CSV = LABELS_DIR / "contacts.csv"

GROUPS = ("high_force", "non_high_force")
GROUP_LABELS = {"high_force": "High-force (force-cluster)", "non_high_force": "Non-high-force"}
GROUP_COLORS = {"high_force": "#D95F02", "non_high_force": "#2474B5"}


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


def load_particles():
    if not PARTICLES_CSV.exists():
        raise FileNotFoundError(f"Run 00_export_workbook_labels.py first: {PARTICLES_CSV}")
    return pd.read_csv(PARTICLES_CSV)


def load_contacts():
    if not CONTACTS_CSV.exists():
        raise FileNotFoundError(f"Run 00_export_workbook_labels.py first: {CONTACTS_CSV}")
    return pd.read_csv(CONTACTS_CSV)


def mean_ci(values):
    """Mean and 95% t interval over systems (the replicates)."""
    from scipy import stats

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan, np.nan
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
