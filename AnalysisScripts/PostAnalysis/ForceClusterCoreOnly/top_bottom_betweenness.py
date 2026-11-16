"""Top-bottom (bottom-to-top percolation) betweenness on the core contact graph.

Standard betweenness counts shortest paths between all particle pairs. Loading percolates
from the bottom wall to the top wall, so the top-bottom betweenness counts only paths that
join a bottom-wall particle to a top-wall particle:

    TB(v) = 1 / (|B| |T|) * sum_{s in B} sum_{t in T, t != v, s != v} sigma_st(v) / sigma_st
    TB(e) = 1 / (|B| |T|) * sum_{s in B} sum_{t in T} sigma_st(e) / sigma_st

B (T) = particles with at least one contact with a bottom (top) wall; sigma_st = number of
shortest s-t paths; sigma_st(v) = those passing through v. Paths use particle-particle contacts
only (walls are not nodes, so no path can shortcut through a wall). A value is the average,
over bottom-top pairs, of the share of that pair's shortest paths through v (or e); endpoints
get no credit for their own pairs, as in standard betweenness. Pairs that are not connected
add nothing. Computed by Brandes accumulation from every bottom particle.
"""

from __future__ import annotations

from collections import deque


def top_bottom_betweenness(adjacency, bottom, top):
    """adjacency: dict node -> iterable of neighbours (undirected). Returns (node_tb, edge_tb,
    connected_pairs) with edge keys (min(u, v), max(u, v)); normalized by |B| |T|."""
    top = set(top)
    bottom = list(bottom)
    node_tb = {v: 0.0 for v in adjacency}
    edge_tb = {}
    connected = 0
    for s in bottom:
        sigma = {s: 1.0}
        dist = {s: 0}
        preds = {s: []}
        order = []
        queue = deque([s])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in adjacency[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    sigma[w] = 0.0
                    preds[w] = []
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        connected += sum(1 for t in top if t in dist and t != s)
        delta = dict.fromkeys(order, 0.0)
        for w in reversed(order):
            flow = (1.0 if (w in top and w != s) else 0.0) + delta[w]
            for v in preds[w]:
                share = sigma[v] / sigma[w] * flow
                key = (v, w) if v < w else (w, v)
                edge_tb[key] = edge_tb.get(key, 0.0) + share
                delta[v] += share
            if w != s:
                node_tb[w] += delta[w]
    pairs = len(bottom) * len(top)
    if pairs:
        node_tb = {v: x / pairs for v, x in node_tb.items()}
        edge_tb = {e: x / pairs for e, x in edge_tb.items()}
    return node_tb, edge_tb, connected


def brute_force(graph, bottom, top):
    """Reference implementation by path enumeration (small graphs only, for checks)."""
    import networkx as nx

    node_tb = {v: 0.0 for v in graph}
    edge_tb = {}
    for s in bottom:
        for t in top:
            if s == t or not nx.has_path(graph, s, t):
                continue
            paths = list(nx.all_shortest_paths(graph, s, t))
            for path in paths:
                for v in path[1:-1]:
                    node_tb[v] += 1 / len(paths)
                for u, v in zip(path[:-1], path[1:]):
                    key = (min(u, v), max(u, v))
                    edge_tb[key] = edge_tb.get(key, 0.0) + 1 / len(paths)
    pairs = len(bottom) * len(top)
    return ({v: x / pairs for v, x in node_tb.items()}, {e: x / pairs for e, x in edge_tb.items()})


def self_test(seed=3):
    import networkx as nx
    import numpy as np

    graph = nx.random_geometric_graph(80, 0.2, seed=seed)
    nodes = sorted(graph)
    bottom, top = nodes[:8], nodes[-9:]
    node_tb, edge_tb, _ = top_bottom_betweenness({v: list(graph[v]) for v in graph}, bottom, top)
    ref_node, ref_edge = brute_force(graph, bottom, top)
    assert np.allclose([node_tb[v] for v in nodes], [ref_node[v] for v in nodes], atol=1e-12)
    keys = set(edge_tb) | set(ref_edge)
    assert np.allclose([edge_tb.get(k, 0) for k in keys], [ref_edge.get(k, 0) for k in keys], atol=1e-12)
    # Node values match networkx's subset betweenness (which halves undirected sums).
    nx_node = nx.betweenness_centrality_subset(graph, bottom, top, normalized=False)
    pairs = len(bottom) * len(top)
    assert np.allclose([node_tb[v] for v in nodes], [2 * nx_node[v] / pairs for v in nodes], atol=1e-12)
    return True


if __name__ == "__main__":
    print("self test passed:", self_test(3) and self_test(11))
