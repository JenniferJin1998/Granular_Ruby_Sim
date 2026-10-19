#!/usr/bin/env python3
"""Exploratory crystal-reference feature matching for periodic particle networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from force_threshold_percolation import atomic_csv, atomic_json, load_graphs, unwrap_component, wall_contact_particles


PROJECT = Path(__file__).resolve().parents[2]
PERIODIC = PROJECT / "AnalysisResults/periodic_boundaries/2026-08-03"
CRYSTALS = PROJECT / "AnalysisResults/crystal_references/0_graph_and_basic_stats/graph_data"
MODELS = {
    "local": ["degree", "q4", "q6", "qbar4", "qbar6"],
    "angular_only": ["q4", "q6", "qbar4", "qbar6"],
    "network_augmented": ["degree", "closeness", "betweenness", "q4", "q6"],
}
STRUCTURES = ["SC", "BCC", "FCC", "HCP"]
COLORS = {"0deg": "#0072B2", "30deg": "#D55E00"}


def save(figure, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(destination.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_vectors(values, references, scales):
    scaled = values / scales
    targets = references / scales
    denominator = np.linalg.norm(scaled, axis=1)[:, None] * np.linalg.norm(targets, axis=1)[None, :]
    cosine = np.divide(scaled @ targets.T, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)
    cosine = np.clip(cosine, -1, 1)
    distance = np.sqrt(np.mean((scaled[:, None, :] - targets[None, :, :]) ** 2, axis=2))
    return cosine, distance


def load_features():
    nodes = pd.read_csv(PERIODIC / "0_graph_and_basic_stats/graph_data/node_features.csv")
    nodes = nodes.loc[~nodes.is_wall].copy()
    bonds = pd.read_csv(PERIODIC / "1_network_property_comparison/bond_order/tables/bond_order_nodes.csv")
    bonds = bonds[bonds.system.str.startswith("ruby_")].copy()
    keys = bonds.system.str.extract(r"ruby_(?P<geometry>\d+deg)_sim(?P<sim_idx>\d+)$")
    if keys.isna().any().any():
        raise ValueError("Unrecognized Ruby bond-order system IDs")
    bonds["geometry"] = keys.geometry
    bonds["sim_idx"] = keys.sim_idx.astype(int)
    properties = ["coordination", "q4", "q6", "qbar4", "qbar6", "mean_s6", "fraction_s6_above_threshold"]
    joined = nodes.merge(bonds[["geometry", "sim_idx", "node_id"] + properties],
                         on=["geometry", "sim_idx", "node_id"], validate="one_to_one", how="left", indicator=True)
    if not joined._merge.eq("both").all() or not np.array_equal(joined.degree, joined.coordination):
        raise ValueError("Particle features and bond-order table do not agree")
    if len(joined) != 63000 or joined.groupby(["geometry", "sim_idx"]).size().ne(1575).any():
        raise ValueError("Expected 40 complete particle systems")
    return joined.drop(columns="_merge"), pd.read_csv(CRYSTALS / "node_features.csv")


def score_model(nodes, crystals, model, output):
    features = MODELS[model]
    reference_rows = []
    reference_nodes = []
    for structure in STRUCTURES:
        frame = crystals[crystals.geometry == structure]
        interior = frame[(frame.boundary_distance_all >= 2) & (frame.degree == frame.degree.max())]
        if interior.empty or not np.isfinite(interior[features].to_numpy()).all():
            raise ValueError(f"No finite bulk reference for {structure}")
        reference_rows.append(dict(structure=structure, bulk_count=len(interior), **interior[features].mean().to_dict()))
        for feature in features:
            reference_nodes.append(dict(model=model, structure=structure, feature=feature,
                                        bulk_std=interior[feature].std(), bulk_min=interior[feature].min(), bulk_max=interior[feature].max()))
    reference = pd.DataFrame(reference_rows).set_index("structure").loc[STRUCTURES]
    targets = reference[features].to_numpy()
    scales = np.sqrt(np.mean(targets ** 2, axis=0))
    if np.any(scales <= 1e-12):
        raise ValueError("A feature has effectively zero reference scale")
    eligible = (nodes.degree >= 4) & np.isfinite(nodes[features].to_numpy()).all(axis=1)
    result = nodes[["geometry", "sim_idx", "node_id", "x", "y", "z", "degree", "closeness", "betweenness", "q4", "q6", "qbar4", "qbar6", "mean_s6", "fraction_s6_above_threshold"]].copy()
    result["eligible"] = eligible
    cosine, distance = compare_vectors(nodes.loc[eligible, features].to_numpy(), targets, scales)
    raw_cosine, _ = compare_vectors(nodes.loc[eligible, features].to_numpy(), targets, np.ones(len(features)))
    for index, structure in enumerate(STRUCTURES):
        result.loc[eligible, f"cosine_{structure}"] = cosine[:, index]
        result.loc[eligible, f"distance_{structure}"] = distance[:, index]
        result.loc[eligible, f"raw_cosine_{structure}"] = raw_cosine[:, index]
    best_distance = np.argmin(distance, axis=1)
    best_cosine = np.argmax(cosine, axis=1)
    result.loc[eligible, "nearest_reference_distance"] = np.asarray(STRUCTURES)[best_distance]
    result.loc[eligible, "nearest_reference_cosine"] = np.asarray(STRUCTURES)[best_cosine]
    result.loc[eligible, "minimum_distance"] = distance.min(axis=1)
    result.loc[eligible, "maximum_cosine"] = cosine.max(axis=1)
    result.loc[eligible, "feature_angle_degrees"] = np.degrees(np.arccos(cosine.max(axis=1)))
    ordered = np.sort(distance, axis=1)
    result.loc[eligible, "distance_margin"] = ordered[:, 1] - ordered[:, 0]
    cutoff = float(result.loc[eligible, "minimum_distance"].quantile(0.05))
    cosine_cutoff = float(result.loc[eligible, "maximum_cosine"].quantile(0.95))
    result["top5_distance"] = eligible & (result.minimum_distance <= cutoff)
    result["top5_cosine"] = eligible & (result.maximum_cosine >= cosine_cutoff)
    result["rank_distance_pooled"] = result.minimum_distance.rank(method="min")
    result["rank_cosine_pooled"] = result.maximum_cosine.rank(ascending=False, method="min")
    result["rank_distance_in_system"] = result.groupby(["geometry", "sim_idx"]).minimum_distance.rank(method="min")
    atomic_csv(output / "tables" / f"{model}_particle_scores.csv", result)
    atomic_csv(output / "tables" / f"{model}_reference_prototypes.csv", reference.reset_index())
    atomic_csv(output / "tables" / f"{model}_feature_scales.csv", pd.DataFrame(dict(feature=features, scale=scales)))
    atomic_csv(output / "tables" / f"{model}_reference_variation.csv", pd.DataFrame(reference_nodes))
    atomic_json(output / "tables" / f"{model}_screening_cutoffs.json", dict(
        distance_top5_cutoff=cutoff, cosine_top5_cutoff=cosine_cutoff, eligible_count=int(eligible.sum()),
        excluded_count=int((~eligible).sum()), threshold_type="pooled relative ranking, not a crystalline-phase threshold"))
    self_cosine, self_distance = compare_vectors(targets, targets, scales)
    if not np.allclose(np.diag(self_cosine), 1) or not np.allclose(np.diag(self_distance), 0):
        raise ValueError("Reference self-match validation failed")
    doubled_cosine, doubled_distance = compare_vectors(2 * targets, targets, scales)
    if not np.allclose(np.diag(doubled_cosine), 1) or not np.all(np.diag(doubled_distance) > 0):
        raise ValueError("Cosine magnitude-degeneracy check failed")
    return result, reference, scales


def plot_scores(models, reference, output):
    local = models["local"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    distance_bins = np.linspace(0, local.minimum_distance.max(), 45)
    cosine_bins = np.linspace(local.maximum_cosine.min(), 1, 45)
    for geometry, color in COLORS.items():
        frame = local[local.geometry == geometry]
        axes[0].hist(frame.minimum_distance.dropna(), bins=distance_bins, density=True, histtype="step", linewidth=2, color=color, label=geometry)
        axes[1].hist(frame.maximum_cosine.dropna(), bins=cosine_bins, density=True, histtype="step", linewidth=2, color=color, label=geometry)
        sample = frame[frame.eligible].sample(min(2000, int(frame.eligible.sum())), random_state=20261018)
        axes[2].scatter(sample.maximum_cosine, sample.minimum_distance, s=5, alpha=0.2, color=color, label=geometry, rasterized=True)
    axes[0].set(xlabel="Minimum scaled RMS distance (lower is closer)", ylabel="Particle probability density")
    axes[1].set(xlabel="Maximum scaled cosine (higher is closer)", ylabel="Particle probability density")
    axes[2].set(xlabel="Maximum scaled cosine", ylabel="Minimum scaled RMS distance")
    for axis in axes:
        axis.legend()
    save(figure, output / "figures/score_distributions_and_cosine_caution")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True)
    for axis, geometry in zip(axes, COLORS):
        sample = local[(local.geometry == geometry) & local.eligible].sample(5000, random_state=20261018)
        artist = axis.scatter(sample.q4, sample.q6, c=sample.minimum_distance, s=5, alpha=0.65,
                              vmin=0, vmax=local.minimum_distance.quantile(0.95), cmap="viridis_r", rasterized=True)
        for structure, row in reference.iterrows():
            axis.scatter(row.q4, row.q6, c="red", marker="x", s=80)
            axis.annotate(structure, (row.q4, row.q6), xytext=(5, 5), textcoords="offset points")
        axis.set(xlabel="q4", ylabel="q6", title=geometry)
    figure.colorbar(artist, ax=axes, label="Local-feature distance (lower is closer)")
    save(figure, output / "figures/q4_q6_with_bulk_references")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    summaries = []
    for geometry, group in local.groupby("geometry"):
        for sim_idx, frame in group.groupby("sim_idx"):
            summaries.append(dict(geometry=geometry, sim_idx=sim_idx,
                                 mean_distance=frame.minimum_distance.mean(),
                                 top5_fraction_all_particles=frame.top5_distance.mean(),
                                 eligible_count=int(frame.eligible.sum())))
    summaries = pd.DataFrame(summaries)
    for index, (geometry, color) in enumerate(COLORS.items()):
        frame = summaries[summaries.geometry == geometry]
        for axis, metric in zip(axes, ["mean_distance", "top5_fraction_all_particles"]):
            axis.scatter(index + np.linspace(-0.1, 0.1, len(frame)), frame[metric], color=color)
    for axis, label in zip(axes, ["Mean local-feature distance (eligible particles)", "Fraction of all particles in pooled top-5% screen"]):
        axis.set(xticks=[0, 1], xticklabels=list(COLORS), ylabel=label)
    save(figure, output / "figures/simulation_level_comparison")
    atomic_csv(output / "tables/simulation_summaries.csv", summaries)


def local_plot(axis, graph, center, box, title, scores=None):
    hops = nx.single_source_shortest_path_length(graph, center, cutoff=2)
    neighborhood = graph.subgraph(hops).copy()
    unwrapped, wraps = unwrap_component(neighborhood, box, [0, 1])
    if any(wraps.values()):
        raise ValueError("Selected two-hop neighborhood wraps a complete periodic cycle")
    origin = unwrapped[center]
    nodes = list(neighborhood)
    positions = {node: (unwrapped[node] - origin) * 1e6 for node in nodes}
    segments = [np.array([positions[first], positions[second]]) for first, second in neighborhood.edges()]
    axis.add_collection3d(Line3DCollection(segments, colors="gray", linewidths=0.65, alpha=0.5))
    xyz = np.array([positions[node] for node in nodes])
    if scores is None:
        artist = axis.scatter(*xyz.T, c="slategray", s=25, depthshade=False)
    else:
        colors = np.asarray([scores.loc[node, "minimum_distance"] for node in nodes])
        missing = ~np.isfinite(colors)
        axis.scatter(*xyz[missing].T, c="lightgray", s=25, depthshade=False)
        artist = axis.scatter(*xyz[~missing].T, c=colors[~missing], cmap="viridis_r", vmin=0, vmax=0.8, s=25, depthshade=False)
    axis.scatter(0, 0, 0, c="red", marker="*", s=110, depthshade=False)
    span = max(np.ptp(xyz, axis=0).max(), 150)
    midpoint = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    axis.set(xlim=(midpoint[0]-span/2, midpoint[0]+span/2), ylim=(midpoint[1]-span/2, midpoint[1]+span/2),
             zlim=(midpoint[2]-span/2, midpoint[2]+span/2), xlabel="x (µm)", ylabel="y (µm)", zlabel="z (µm)", title=title)
    axis.set_box_aspect((1, 1, 1))
    return artist, hops


def spatial_analysis(local, crystal_features, reference, scales, output, reuse_spatial=False):
    graphs = load_graphs(str(PERIODIC / "0_graph_and_basic_stats/graph_data/graph_dict_labeled.pkl"))
    reference_graphs = load_graphs(str(CRYSTALS / "graph_dict_labeled.pkl"))
    if not reuse_spatial:
        local["boundary_hops"] = np.nan
        local["neighbor_top5_fraction"] = np.nan
        local["neighbor_same_reference_fraction"] = np.nan
    regions = []
    for (geometry, sim_idx), frame in local.groupby(["geometry", "sim_idx"]):
        if reuse_spatial:
            continue
        core = graphs[geometry]["core"][int(sim_idx)]
        full = graphs[geometry]["full"][int(sim_idx)]
        boundary = wall_contact_particles(full, [-1, -2, -3, -4, -5])
        hops = nx.multi_source_dijkstra_path_length(core, boundary, weight=None)
        indexed = frame.set_index("node_id")
        top_nodes = set(indexed.index[indexed.top5_distance])
        for row in frame.itertuples():
            neighbors = list(core.neighbors(row.node_id))
            local.loc[row.Index, "boundary_hops"] = hops.get(row.node_id, np.nan)
            if neighbors:
                local.loc[row.Index, "neighbor_top5_fraction"] = len(set(neighbors) & top_nodes) / len(neighbors)
                if row.eligible:
                    local.loc[row.Index, "neighbor_same_reference_fraction"] = indexed.loc[neighbors, "nearest_reference_distance"].eq(row.nearest_reference_distance).mean()
        components = sorted(nx.connected_components(core.subgraph(top_nodes)), key=lambda nodes: (-len(nodes), min(nodes)))
        for component_id, nodes in enumerate(components):
            region = indexed.loc[list(nodes)]
            regions.append(dict(geometry=geometry, sim_idx=sim_idx, component_id=component_id,
                                node_count=len(nodes), mean_distance=region.minimum_distance.mean(),
                                mean_s6=region.mean_s6.mean(), node_ids=";".join(map(str, sorted(nodes)))))
    atomic_csv(output / "tables/local_particle_scores.csv", local)
    if not reuse_spatial:
        atomic_csv(output / "tables/top5_connected_regions.csv", pd.DataFrame(regions))
    candidate_rows, neighbor_rows = [], []
    for geometry in COLORS:
        for mode, ranking, ascending in (("distance", "minimum_distance", True), ("cosine", "maximum_cosine", False)):
            candidates = local[(local.geometry == geometry) & local.eligible].sort_values([ranking, "sim_idx", "node_id"], ascending=[ascending, True, True])
            candidates = candidates.drop_duplicates("sim_idx").head(3)
            for rank, candidate in enumerate(candidates.itertuples(), 1):
                structure = getattr(candidate, f"nearest_reference_{mode}")
                core = graphs[geometry]["core"][candidate.sim_idx]
                ideal = reference_graphs[structure]["core"][0]
                reference_nodes = crystal_features[(crystal_features.geometry == structure) &
                                                  (crystal_features.boundary_distance_all >= 2) &
                                                  (crystal_features.degree == crystal_features[crystal_features.geometry == structure].degree.max())]
                ideal_center = int(reference_nodes.sort_values(["boundary_distance_all", "node_id"], ascending=[False, True]).iloc[0].node_id)
                scores = local[(local.geometry == geometry) & (local.sim_idx == candidate.sim_idx)].set_index("node_id")
                figure = plt.figure(figsize=(16, 5), constrained_layout=True)
                first = figure.add_subplot(131, projection="3d")
                artist, hops = local_plot(first, core, candidate.node_id, {0: .0018, 1: .003},
                                         f"{geometry}, sim{candidate.sim_idx:03d}, node {candidate.node_id}\nTwo-hop contact neighborhood", scores)
                second = figure.add_subplot(132, projection="3d")
                cell = np.asarray(ideal.graph["cell_matrix"])
                local_plot(second, ideal, ideal_center, {0: cell[0, 0], 1: cell[1, 1]}, f"{structure}: ideal bulk two-hop neighborhood")
                third = figure.add_subplot(133)
                features = MODELS["local"]
                values = np.array([getattr(candidate, feature) for feature in features]) / scales
                third.plot(features, values, "o-", label="Candidate")
                third.plot(features, reference.loc[structure, features].to_numpy() / scales, "s--", label=f"{structure} prototype")
                third.set(ylabel="Feature / common reference RMS scale", title=f"Selected by {mode}\ncos={getattr(candidate, 'cosine_' + structure):.4f}, d={getattr(candidate, 'distance_' + structure):.4f}")
                third.legend()
                figure.colorbar(artist, ax=[first, second], shrink=0.6, label="Ruby local-feature distance (gray: unscored)")
                save(figure, output / "neighborhoods" / f"{geometry}_{mode}_rank{rank}_sim{candidate.sim_idx:03d}_node{candidate.node_id}")
                candidate_rows.append(dict(geometry=geometry, sim_idx=candidate.sim_idx, node_id=candidate.node_id,
                                           selection=mode, rank=rank, reference=structure, minimum_distance=candidate.minimum_distance,
                                           maximum_cosine=candidate.maximum_cosine, mean_s6=candidate.mean_s6,
                                           boundary_hops=candidate.boundary_hops, neighbor_count=len(hops)))
                for node, hop in hops.items():
                    neighbor_rows.append(dict(geometry=geometry, sim_idx=candidate.sim_idx, center_node_id=candidate.node_id,
                                              selection=mode, node_id=node, hop=hop, minimum_distance=scores.loc[node, "minimum_distance"]))
        best = local[(local.geometry == geometry) & local.eligible].sort_values("minimum_distance").iloc[0]
        sample = local[(local.geometry == geometry) & (local.sim_idx == best.sim_idx)]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for axis, horizontal in zip(axes, ["x", "y"]):
            axis.scatter(sample[horizontal]*1000, sample.z*1000, c="lightgray", s=8)
            artist = axis.scatter(sample.loc[sample.eligible, horizontal]*1000, sample.loc[sample.eligible, "z"]*1000,
                                  c=sample.loc[sample.eligible, "minimum_distance"], s=10, cmap="viridis_r", vmin=0, vmax=0.8)
            selected = sample[sample.top5_distance]
            axis.scatter(selected[horizontal]*1000, selected.z*1000, s=25, facecolors="none", edgecolors="red", linewidths=0.7)
            axis.set(xlabel=f"{horizontal} (mm)", ylabel="z (mm)", title=f"{geometry}, sim{int(best.sim_idx):03d}; red rings = pooled top 5%")
            axis.set_aspect("equal")
        figure.colorbar(artist, ax=axes, label="Minimum local-feature distance")
        save(figure, output / "figures" / f"{geometry}_representative_spatial_map")
    atomic_csv(output / "tables/visualized_candidates.csv", pd.DataFrame(candidate_rows))
    atomic_csv(output / "tables/visualized_neighborhood_members.csv", pd.DataFrame(neighbor_rows))
    return local


def write_results(local, output):
    lines = ["# Method 1: initial screening results", "",
             "These are contact-environment similarity rankings, not crystalline-phase identifications.", "",
             f"Joined {len(local):,} particles. Scored {int(local.eligible.sum()):,}; excluded {int((~local.eligible).sum()):,} for degree <4 or nonfinite features.", "",
             "| Geometry | Mean distance, eligible | Top-distance candidates | Candidate mean s6 | All eligible mean s6 | Candidate mean boundary hops |",
             "|---|---:|---:|---:|---:|---:|"]
    for geometry, frame in local.groupby("geometry"):
        candidates = frame[frame.top5_distance]
        eligible = frame[frame.eligible]
        lines.append(f"| {geometry} | {eligible.minimum_distance.mean():.4f} | {len(candidates)} | {candidates.mean_s6.mean():.4f} | {eligible.mean_s6.mean():.4f} | {candidates.boundary_hops.mean():.2f} |")
    lines.extend(["", "The pooled lowest 5% distance screen is common to both geometries; the reported counts are selected candidates, NOT crystalline fractions. Bond-coherence s6 was not included in the feature vectors, so enrichment in s6 is a useful complementary check, not independent confirmation of a phase.", "",
                  "## Model sensitivity", "", "| Model | Distance-rank correlation with local model | Top-5% set Jaccard overlap with local model | Cosine/distance rank correlation |", "|---|---:|---:|---:|"])
    sensitivity = pd.read_csv(output / "tables/model_sensitivity.csv")
    for row in sensitivity.itertuples():
        lines.append(f"| {row.model} | {row.distance_rank_spearman_with_local:.3f} | {row.top5_jaccard_with_local:.3f} | {row.cosine_vs_distance_rank_spearman:.3f} |")
    lines.extend(["", "Low overlap means candidate identification is sensitive to feature choice; do not interpret one model's nearest crystal label as a robust phase assignment.", "",
                  "## Inspect first", "", "- `figures/score_distributions_and_cosine_caution`: cosine and magnitude-sensitive distance are not interchangeable.",
                  "- `neighborhoods/`: two-hop candidates selected by distance and separately by cosine, with ideal references and feature profiles.",
                  "- `tables/visualized_candidates.csv`: exact simulation/particle IDs and held-out bond coherence.",
                  "- `tables/top5_connected_regions.csv`: connected groups of candidates, not necessarily a single crystal-like motif.",
                  "- `README.md`: scaling, bulk reference selection, boundary/neighbor-definition caveats, and every figure/table explained."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run(output):
    nodes, crystals = load_features()
    models, references, scales = {}, {}, {}
    for model in MODELS:
        models[model], references[model], scales[model] = score_model(nodes, crystals, model, output)
    plot_scores(models, references["local"], output)
    print("Scored all particles; preparing spatial neighborhoods.", flush=True)
    models["local"] = spatial_analysis(models["local"], crystals, references["local"], scales["local"], output)
    comparisons = []
    for model in MODELS:
        frame = models[model]
        eligible = frame.eligible & models["local"].eligible
        first = set(frame.index[frame.top5_distance])
        second = set(models["local"].index[models["local"].top5_distance])
        comparisons.append(dict(model=model,
                                distance_rank_spearman_with_local=stats.spearmanr(frame.loc[eligible, "minimum_distance"], models["local"].loc[eligible, "minimum_distance"]).statistic,
                                top5_jaccard_with_local=len(first & second) / len(first | second),
                                cosine_vs_distance_rank_spearman=stats.spearmanr(frame.loc[eligible, "maximum_cosine"], -frame.loc[eligible, "minimum_distance"]).statistic))
    atomic_csv(output / "tables/model_sensitivity.csv", pd.DataFrame(comparisons))
    validation = []
    for (geometry, sim_idx), group in models["local"].groupby(["geometry", "sim_idx"]):
        for population, subset in (("all_eligible", group[group.eligible]), ("top5_distance", group[group.top5_distance]), ("top5_cosine", group[group.top5_cosine])):
            validation.append(dict(geometry=geometry, sim_idx=sim_idx, population=population, count=len(subset),
                                   mean_s6=subset.mean_s6.mean(), fraction_s6_above_threshold=subset.fraction_s6_above_threshold.mean(),
                                   mean_boundary_hops=subset.boundary_hops.mean(), mean_neighbor_top5_fraction=subset.neighbor_top5_fraction.mean()))
    atomic_csv(output / "tables/independent_order_and_boundary_checks.csv", pd.DataFrame(validation))
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(Path(__file__).with_suffix(".md").read_text())
    write_results(models["local"], output)
    atomic_json(output / "validation.json", dict(particle_rows=len(nodes), geometry_count=2, simulation_count=40,
                                                join_one_to_one=True, degree_matches_bond_coordination=True,
                                                reference_self_matches_pass=True, cosine_magnitude_counterexample_pass=True,
                                                feature_models=MODELS, minimum_eligible_degree=4,
                                                classification_status="Exploratory rankings only; not calibrated phase detection",
                                                particle_source=str(PERIODIC / "0_graph_and_basic_stats/graph_data/node_features.csv"),
                                                bond_source=str(PERIODIC / "1_network_property_comparison/bond_order/tables/bond_order_nodes.csv"),
                                                crystal_source=str(CRYSTALS / "node_features.csv")))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PERIODIC / "4_ordered_structure_detection/method_1_feature_similarity")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"], "font.size": 11, "svg.fonttype": "none"})
    run(args.output)
