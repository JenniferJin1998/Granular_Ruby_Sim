#!/usr/bin/env python3
"""Low-dimensional embeddings of graph features: do high-force particles/contacts cluster?

Particles and particle-particle contacts are described by core-graph (no-wall)
features only; nothing derived from force, stress, strain, or deformation, and no
position. Features are nested sets, from the simplest to the full list. Within each
angle (20 systems pooled) every feature is mapped to a normal distribution by rank
(heavy tails and discrete values otherwise dominate the distances), then embedded
with PCA and t-SNE and colored by the workbook label.

Separability is quantified without the embedding: cross-validated ROC AUC of a
logistic regression (linear) and a gradient-boosted tree model (nonlinear) on the
full feature set, with folds split by system so no system is in both train and test.
AUC 0.5 = the features carry no information about the label.

Needs scikit-learn (the gnn_granular environment) and the contact table of step 05.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import QuantileTransformer

from core_only_common import (GROUP_COLORS, GROUP_LABELS, NODE_FEATURES, OUTPUT_ROOT, angle_colors,
                              load_particles, sorted_angles)

OUT = OUTPUT_ROOT / "06_feature_embedding"
EDGE_FEATURES = NODE_FEATURES.parent / "edge_features.csv"
EDGE_BETWEENNESS = OUTPUT_ROOT / "05_correlation" / "tables" / "contacts_edge_betweenness.csv"

NODE_BASIC = ["degree", "closeness", "betweenness", "clustering"]
NODE_EXTRA = ["avg_neighbor_degree", "avg_curvature_no_walls", "principal_eigenvector", "nfd"]
NODE_SETS = {
    "N1_basic": NODE_BASIC,
    "N2_local": NODE_BASIC + ["avg_neighbor_degree", "avg_curvature_no_walls"],
    "N3_all": NODE_BASIC + NODE_EXTRA,
}
EDGE_OWN = ["angle_with_zz", "curvature_no_walls"]
EDGE_GRAPH = ["edge_connectivity", "node_connectivity", "edge_betweenness"]
EDGE_SETS = {
    "E1_geometry": EDGE_OWN,
    "E2_edge_graph": EDGE_OWN + EDGE_GRAPH,
    "E3_plus_node_basic": EDGE_OWN + EDGE_GRAPH + [f"mean_{c}" for c in NODE_BASIC],
    "E4_all": EDGE_OWN + EDGE_GRAPH + [f"mean_{c}" for c in NODE_BASIC + NODE_EXTRA],
}
SEED = 20261103


def load_nodes():
    particles = load_particles()
    features = pd.read_csv(NODE_FEATURES, usecols=["geometry", "sim_idx", "node_id", "is_wall"] + NODE_BASIC
                           + NODE_EXTRA).rename(columns={"geometry": "angle"})
    features = features[~features.is_wall.astype(bool)].drop(columns="is_wall")
    shared = [c for c in NODE_BASIC if c in particles.columns]  # the workbook has no clustering
    nodes = particles[["angle", "sim_idx", "node_id", "group"] + shared].merge(
        features, on=["angle", "sim_idx", "node_id"], how="left", validate="one_to_one", suffixes=("", "_nf"))
    for column in shared:
        if not np.allclose(nodes[column], nodes[f"{column}_nf"], rtol=1e-9, atol=1e-12):
            raise ValueError(f"Workbook {column} differs from node_features.csv")
    nodes = nodes.drop(columns=[f"{c}_nf" for c in shared])
    nodes["label"] = (nodes.group == "high_force").astype(int)
    return nodes


def load_edges(nodes):
    if not EDGE_BETWEENNESS.exists():
        raise FileNotFoundError(f"Run 05_correlation.py first: {EDGE_BETWEENNESS}")
    edges = pd.read_csv(EDGE_BETWEENNESS, usecols=["angle", "sim_idx", "edge_u", "edge_v", "is_high_force",
                                                   "angle_with_zz", "edge_betweenness"])
    graph = pd.read_csv(EDGE_FEATURES, usecols=["geometry", "sim_idx", "node1", "node2", "is_core_edge",
                                                "curvature_no_walls", "edge_connectivity",
                                                "node_connectivity"]).rename(columns={"geometry": "angle"})
    graph = graph[graph.is_core_edge.astype(bool)].drop(columns="is_core_edge")
    graph[["node1", "node2"]] = graph[["node1", "node2"]].astype(int)  # read as text: wall ids are names
    for frame, (a, b) in ((edges, ("edge_u", "edge_v")), (graph, ("node1", "node2"))):
        frame["lo"], frame["hi"] = np.minimum(frame[a], frame[b]), np.maximum(frame[a], frame[b])
    edges = edges.merge(graph.drop(columns=["node1", "node2"]), on=["angle", "sim_idx", "lo", "hi"],
                        how="left", validate="one_to_one")
    if edges[EDGE_GRAPH[:2] + ["curvature_no_walls"]].isna().any().any():
        raise ValueError("Contacts without core edge features in edge_features.csv")
    node_index = nodes.set_index(["angle", "sim_idx", "node_id"])
    for column in NODE_BASIC + NODE_EXTRA:
        ends = [node_index[column].reindex(pd.MultiIndex.from_arrays([edges.angle, edges.sim_idx, edges[e]]))
                .to_numpy(float) for e in ("edge_u", "edge_v")]
        edges[f"mean_{column}"] = 0.5 * (ends[0] + ends[1])
    edges["label"] = edges.is_high_force.astype(bool).astype(int)
    return edges


def transform(frame, columns):
    values = frame[columns].to_numpy(float)
    return QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                               random_state=SEED).fit_transform(values)


def separability(x, y, groups):
    scores = {"logistic": [], "boosted_trees": []}
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        models = {"logistic": LogisticRegression(max_iter=2000),
                  "boosted_trees": HistGradientBoostingClassifier(random_state=SEED)}
        for name, model in models.items():
            model.fit(x[train], y[train])
            scores[name].append(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    return {name: (float(np.mean(v)), float(np.std(v, ddof=1))) for name, v in scores.items()}


def share_map(ax, xy, y, share):
    hb = ax.hexbin(xy[:, 0], xy[:, 1], C=y, gridsize=30, reduce_C_function=np.mean, mincnt=8,
                   cmap="coolwarm", vmin=max(0.0, share - 0.4), vmax=min(1.0, share + 0.4))
    return hb


def plot_set(element, set_name, columns, embeddings, angles, path):
    fig, axes = plt.subplots(3, len(angles), figsize=(4.2 * len(angles), 12), squeeze=False)
    rng = np.random.default_rng(SEED)
    for col, angle in enumerate(angles):
        e = embeddings[angle]
        y, share = e["y_plot"], e["share"]
        for row, (key, name) in enumerate((("pca", "PCA"), ("tsne", "t-SNE"))):
            ax = axes[row, col]
            xy = e[key]
            order = rng.permutation(len(y))
            ax.scatter(xy[order, 0], xy[order, 1], s=2, linewidths=0, alpha=0.45,
                       c=np.where(y[order] == 1, GROUP_COLORS["high_force"], GROUP_COLORS["non_high_force"]))
            ax.set_title(f"{angle} {name}" + (f" ({e['pca_var'][0]:.0%}, {e['pca_var'][1]:.0%} var.)"
                                               if key == "pca" else ""), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = axes[2, col]
        hb = share_map(ax, e["tsne"], y, share)
        ax.set_title(f"{angle} t-SNE: high-force fraction (share {share:.2f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=GROUP_COLORS[g], label=GROUP_LABELS[g])
               for g in ("high_force", "non_high_force")]
    axes[0, -1].legend(handles=handles, loc="upper right", fontsize=8, markerscale=1.5)
    noun = "particles" if element == "nodes" else "contacts"
    fig.suptitle(f"{noun.capitalize()}, feature set {set_name}: {', '.join(columns)}\n"
                 f"Row 3: mean label in hexagons with at least 8 {noun} (red = above the angle's share)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_separability(table, angles, path):
    colors = angle_colors(angles)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), squeeze=False)
    for ax, element in zip(axes[0], ("nodes", "edges")):
        sub = table[table.element == element]
        sets = list(dict.fromkeys(sub.feature_set))
        x = np.arange(len(sets))
        for i, angle in enumerate(angles):
            for model, marker, style in (("logistic", "s", "--"), ("boosted_trees", "o", "-")):
                rows = sub[(sub.angle == angle)].set_index("feature_set").reindex(sets)
                ax.errorbar(x + (i - 1.5) * 0.06, rows[f"auc_{model}"], yerr=rows[f"auc_{model}_sd"],
                            color=colors[angle], marker=marker, linestyle=style, capsize=2,
                            label=f"{angle}, {model.replace('_', ' ')}")
        ax.axhline(0.5, color="k", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sets, rotation=15)
        ax.set_ylabel("ROC AUC (5-fold, split by system)")
        ax.set_title("Particles" if element == "nodes" else "Contacts")
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.suptitle("How well graph features alone separate high-force from non-high-force (0.5 = not at all)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tsne-sample", type=int, default=8000,
                        help="Points per angle passed to t-SNE and plotted (random sample).")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    nodes = load_nodes()
    edges = load_edges(nodes)
    angles = sorted_angles(nodes.angle)
    tables, figures = OUT / "tables", OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    rows, loadings = [], []
    for element, frame, sets in (("nodes", nodes, NODE_SETS), ("edges", edges, EDGE_SETS)):
        (figures / element).mkdir(parents=True, exist_ok=True)
        for set_name, columns in sets.items():
            start = time.time()
            embeddings, coords = {}, []
            for angle in angles:
                sub = frame[frame.angle == angle].dropna(subset=columns).reset_index(drop=True)
                x = transform(sub, columns)
                y = sub.label.to_numpy()
                auc = separability(x, y, sub.sim_idx.to_numpy())
                pca = PCA(n_components=min(len(columns), 10), random_state=SEED).fit(x)
                for k, component in enumerate(pca.components_[:2]):
                    loadings.append({"element": element, "feature_set": set_name, "angle": angle,
                                     "component": f"PC{k + 1}", "explained_variance": pca.explained_variance_ratio_[k],
                                     **dict(zip(columns, component))})
                rng = np.random.default_rng(SEED)
                pick = np.sort(rng.choice(len(sub), size=min(args.tsne_sample, len(sub)), replace=False))
                tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=SEED,
                            n_jobs=args.n_jobs).fit_transform(x[pick])
                pcs = pca.transform(x[pick])[:, :2]
                if pcs.shape[1] == 1:
                    pcs = np.column_stack([pcs, np.zeros(len(pcs))])
                embeddings[angle] = {"pca": pcs, "tsne": tsne, "y_plot": y[pick], "share": y.mean(),
                                     "pca_var": np.append(pca.explained_variance_ratio_, 0.0)}
                coords.append(sub.loc[pick, ["angle", "sim_idx"] + (["node_id"] if element == "nodes"
                                                                       else ["edge_u", "edge_v"]) + ["label"]]
                              .assign(pc1=pcs[:, 0], pc2=pcs[:, 1], tsne1=tsne[:, 0], tsne2=tsne[:, 1]))
                rows.append({"element": element, "feature_set": set_name, "features": " ".join(columns),
                             "angle": angle, "n": len(sub), "high_force_share": y.mean(),
                             "auc_logistic": auc["logistic"][0], "auc_logistic_sd": auc["logistic"][1],
                             "auc_boosted_trees": auc["boosted_trees"][0],
                             "auc_boosted_trees_sd": auc["boosted_trees"][1]})
            plot_set(element, set_name, columns, embeddings, angles, figures / element / f"{set_name}.png")
            pd.concat(coords).to_csv(tables / f"embedding_{element}_{set_name}.csv.gz", index=False)
            print(f"{element} {set_name}: {time.time() - start:.0f} s", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(tables / "separability_auc.csv", index=False)
    pd.DataFrame(loadings).to_csv(tables / "pca_loadings.csv", index=False)
    plot_separability(table, angles, figures / "separability_auc.png")
    (OUT / "metadata.json").write_text(json.dumps({
        "inputs": [str(NODE_FEATURES), str(EDGE_FEATURES), str(EDGE_BETWEENNESS)],
        "labels": "workbook: particle high-force = has a high-force particle-particle contact; "
                  "contact high-force = is_high_force (normal force >= force_split2 threshold)",
        "node_sets": NODE_SETS, "edge_sets": EDGE_SETS,
        "excluded": "force, stress, strain, deformation (delta), positions, contact normals; fiedler "
                    "(sign arbitrary per system, a spatial coordinate); with-wall variants",
        "edge_mean_features": "mean of the two end particles' node features",
        "scaling": "per angle, rank-based map of each feature to a standard normal (QuantileTransformer)",
        "tsne": {"perplexity": args.perplexity, "sample_per_angle": args.tsne_sample, "init": "pca",
                 "random_state": SEED},
        "separability": "ROC AUC, 5-fold GroupKFold by sim_idx; logistic regression and "
                        "HistGradientBoostingClassifier on the transformed full feature set",
    }, indent=2))
    print(table[["element", "feature_set", "angle", "auc_logistic", "auc_boosted_trees"]].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
