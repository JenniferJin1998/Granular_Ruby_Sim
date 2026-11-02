#!/usr/bin/env python3
"""Betweenness rank of each particle within its own contact neighborhood.

Neighborhoods come from the workbook's particle-particle contacts only. For each
particle, the local percentile is the share of particles within graph distance
<= h (h = 1, 2; the particle itself excluded) with lower betweenness, ties counted
half: 1 = higher than all neighbors, 0.5 = typical of its neighbors. A first-shell
maximum is a particle strictly above all of its direct contacts. Systems are the
replicates: per-system means, summarized by mean and 95% t interval per angle.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from core_only_common import (CONTACTS_CSV, GROUP_COLORS, GROUP_LABELS, GROUPS, OUTPUT_ROOT,
                              PARTICLES_CSV, load_contacts, load_particles, mean_ci, sorted_angles,
                              summarize)

OUT = OUTPUT_ROOT / "03_local_betweenness_rank"
HOPS = (1, 2)


def local_ranks(particles, contacts):
    rows = []
    contact_groups = dict(tuple(contacts.groupby(["angle", "sim_idx"])))
    for (angle, sim), sub in particles.groupby(["angle", "sim_idx"]):
        graph = nx.Graph()
        graph.add_nodes_from(sub.node_id)
        edges = contact_groups[angle, sim]
        graph.add_edges_from(zip(edges.edge_u, edges.edge_v))
        values = dict(zip(sub.node_id, sub.betweenness))
        for node in sub.node_id:
            row = {"angle": angle, "sim_idx": sim, "node_id": node}
            distances = nx.single_source_shortest_path_length(graph, node, cutoff=max(HOPS))
            for h in HOPS:
                shell = np.array([values[n] for n, d in distances.items() if 0 < d <= h])
                row[f"local_percentile_hop{h}"] = (
                    ((shell < values[node]).sum() + 0.5 * (shell == values[node]).sum()) / len(shell)
                    if len(shell) else np.nan)
            first = [values[n] for n in graph[node]]
            row["is_local_max_hop1"] = float(values[node] > max(first)) if first else np.nan
            rows.append(row)
    return particles.merge(pd.DataFrame(rows), on=["angle", "sim_idx", "node_id"], validate="one_to_one")


def per_system(local):
    rows = []
    for (angle, sim, group), sub in local.groupby(["angle", "sim_idx", "group"]):
        rows.append({"angle": angle, "sim_idx": sim, "group": group, "particles": len(sub),
                     **{f"mean_local_percentile_hop{h}": sub[f"local_percentile_hop{h}"].mean() for h in HOPS},
                     "fraction_local_max_hop1": sub.is_local_max_hop1.mean()})
    return pd.DataFrame(rows)


def plot_per_system(systems, angles, path):
    metrics = [(f"mean_local_percentile_hop{h}", f"Local betweenness percentile, hop <= {h}", 0.5)
               for h in HOPS]
    metrics.append(("fraction_local_max_hop1", "Fraction that are first-shell maxima", None))
    x = np.arange(len(angles))
    offsets = dict(zip(GROUPS, (-0.12, 0.12)))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.4), squeeze=False)
    for ax, (column, title, reference) in zip(axes[0], metrics):
        for group in GROUPS:
            means = []
            for i, angle in enumerate(angles):
                values = systems[(systems.angle == angle) & (systems.group == group)][column]
                ax.scatter(i + offsets[group] + rng.uniform(-0.03, 0.03, len(values)), values, s=8,
                           color=GROUP_COLORS[group], alpha=0.35, linewidths=0)
                means.append(mean_ci(values))
            means = np.array(means)
            ax.errorbar(x + offsets[group], means[:, 0],
                        yerr=[means[:, 0] - means[:, 1], means[:, 2] - means[:, 0]],
                        color=GROUP_COLORS[group], marker="o", capsize=3, linewidth=1.5,
                        label=GROUP_LABELS[group])
        if reference is not None:
            ax.axhline(reference, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(angles)
        ax.set_title(title)
        ax.set_ylabel("Per-system mean")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Local betweenness rank (dots = systems; line = mean with 95% CI; 0.5 = typical of neighbors)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_ecdf(local, angles, path):
    fig, axes = plt.subplots(1, len(angles), figsize=(4.4 * len(angles), 4.0), sharey=True, squeeze=False)
    for ax, angle in zip(axes[0], angles):
        for group in GROUPS:
            values = np.sort(local[(local.angle == angle) & (local.group == group)]
                             .local_percentile_hop1.dropna().to_numpy())
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                    color=GROUP_COLORS[group], label=f"{GROUP_LABELS[group]} (n={len(values):,})")
        ax.set_title(angle)
        ax.set_xlabel("Local betweenness percentile among first neighbors")
        ax.legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("Cumulative fraction of particles")
    fig.suptitle("First-shell local betweenness percentile (pooled particles; core only)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    particles = load_particles()
    contacts = load_contacts()
    angles = sorted_angles(particles.angle)
    local = local_ranks(particles, contacts)
    systems = per_system(local)

    tables, figures = OUT / "tables", OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    local.to_csv(tables / "particles_local_rank.csv", index=False)
    systems.to_csv(tables / "local_rank_per_system.csv", index=False)
    columns = [f"mean_local_percentile_hop{h}" for h in HOPS] + ["fraction_local_max_hop1"]
    summary = summarize(systems, ["angle", "group"], columns)
    summary.to_csv(tables / "local_rank_summary.csv", index=False)

    plot_per_system(systems, angles, figures / "local_rank_per_system.png")
    plot_ecdf(local, angles, figures / "local_rank_ecdf_hop1.png")
    (OUT / "metadata.json").write_text(json.dumps({
        "inputs": [str(PARTICLES_CSV), str(CONTACTS_CSV)],
        "neighborhood": "particle-particle contacts only; hop <= h, particle itself excluded",
        "local_percentile": "share of neighborhood with lower betweenness, ties counted 1/2",
        "isolated_particles": int(local.local_percentile_hop1.isna().sum()),
        "replicates": "systems; mean and 95% t interval over systems per angle",
    }, indent=2))
    print(summary.pivot(index="angle", columns="group", values=f"{columns[0]}_mean").round(3).to_string())
    print(summary.pivot(index="angle", columns="group", values=f"{columns[2]}_mean").round(4).to_string())


if __name__ == "__main__":
    main()
