"""
graph3d_viz.py
==============
3-D visualisation of particle contact-network graphs stored in the
graph_dict format produced by GraphGeneration.py.

Data layout
-----------
    GL_final[angle_label]['full' | 'core'][sim_index]
                              ↑ NetworkX Graph

Node attributes (particle nodes)
    position         : (x, y, z)
    is_wall          : False
    in_center_region : bool   True → cylinder-centre particle
    stress_vm, stress_hydro, stress_11 … stress_12
    degree, closeness, betweenness, clustering
    degree_with_walls, …, avg_curvature_with_walls, avg_curvature_no_walls
    high_force_degree, is_force_chain_node, force_chain_role
    hf_cluster_id, hf_cluster_type   (if clustering ran)

Node attributes (wall nodes – full graph only)
    is_wall      : True
    wall_label   : int (negative pid)
    No 'position' key; position derived from adjacent edge contact_location.

Edge attributes
    contact_location  : (cx, cy, cz)
    normal_force      : float
    tangential_force  : float
    is_high_force     : bool
    angle_with_zz     : float
    is_wall_contact   : bool
    curvature_with_walls, curvature_no_walls   (if Ricci ran)
    hf_cluster_id, hf_cluster_type             (if clustering ran)

Quick start
-----------
    from graph3d_viz import plot_graph_3d

    # Random graph for each geometry, core view, all gray
    plot_graph_3d(GL_final)

    # Specific simulations, colored by stress Von Mises
    plot_graph_3d(GL_final,
                  geometries=['0deg', '45deg'], sim_indices=[3, 17],
                  node_attr='stress_vm', edge_attr='normal_force')

    # Full graph (wall nodes shown as cubes), transparent background, save
    plot_graph_3d(GL_final, graph_view='full',
                  node_attr='high_force_degree',
                  bg='transparent', save_path='out.png')
"""

import os
import random
import warnings
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# ---------------------------------------------------------------------------
# Parula colormap  (256-entry, faithful to MATLAB's default colourmap)
# ---------------------------------------------------------------------------
_PARULA_DATA = np.array([
    [0.2081, 0.1663, 0.5292], [0.2116, 0.1898, 0.5777], [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771], [0.1959, 0.2645, 0.7279], [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303], [0.0591, 0.3598, 0.8683], [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828], [0.0165, 0.4266, 0.8786], [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641], [0.0629, 0.4737, 0.8554], [0.0723, 0.4887, 0.8467],
    [0.0780, 0.5040, 0.8384], [0.0793, 0.5200, 0.8312], [0.0749, 0.5375, 0.8264],
    [0.0641, 0.5569, 0.8239], [0.0488, 0.5772, 0.8228], [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135], [0.0239, 0.6287, 0.8038], [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768], [0.0267, 0.6642, 0.7607], [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254], [0.0843, 0.6928, 0.7068], [0.1133, 0.7015, 0.6877],
    [0.1453, 0.7098, 0.6683], [0.1801, 0.7177, 0.6486], [0.2178, 0.7250, 0.6287],
    [0.2586, 0.7317, 0.6083], [0.3022, 0.7376, 0.5875], [0.3482, 0.7424, 0.5660],
    [0.3953, 0.7459, 0.5436], [0.4420, 0.7481, 0.5202], [0.4871, 0.7491, 0.4958],
    [0.5300, 0.7491, 0.4702], [0.5709, 0.7485, 0.4436], [0.6099, 0.7473, 0.4160],
    [0.6473, 0.7456, 0.3876], [0.6834, 0.7435, 0.3584], [0.7184, 0.7411, 0.3286],
    [0.7525, 0.7384, 0.2984], [0.7858, 0.7356, 0.2680], [0.8185, 0.7327, 0.2373],
    [0.8507, 0.7299, 0.2067], [0.8824, 0.7274, 0.1760], [0.9139, 0.7258, 0.1453],
    [0.9450, 0.7261, 0.1142], [0.9739, 0.7314, 0.0812], [0.9938, 0.7455, 0.0539],
    [0.9990, 0.7653, 0.0438], [0.9955, 0.7861, 0.0633], [0.9880, 0.8071, 0.0911],
    [0.9828, 0.8282, 0.1207], [0.9805, 0.8492, 0.1515], [0.9828, 0.8700, 0.1821],
    [0.9907, 0.8901, 0.2120], [0.9999, 0.9093, 0.2395], [0.9999, 0.9274, 0.2642],
    [0.9999, 0.9445, 0.2865],
], dtype=float)

# Upsample to 256 via linear interpolation
def _make_parula_cmap():
    n_in = len(_PARULA_DATA)
    n_out = 256
    t_in = np.linspace(0, 1, n_in)
    t_out = np.linspace(0, 1, n_out)
    rgb256 = np.column_stack([np.interp(t_out, t_in, _PARULA_DATA[:, ch]) for ch in range(3)])
    return mcolors.ListedColormap(rgb256, name='parula')

PARULA = _make_parula_cmap()
matplotlib.colormaps.register(PARULA, name='parula', force=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cmap(name):
    """Return a matplotlib Colormap given a name (supports 'parula')."""
    if name == 'parula':
        return PARULA
    return mcm.get_cmap(name)


def _wall_position(G, wall_node):
    """
    Estimate the 3-D position of a wall node from adjacent edge contact_locations.
    Returns None if no contact_location is available.
    """
    locs = []
    for nbr in G.neighbors(wall_node):
        edata = G[wall_node][nbr]
        if 'contact_location' in edata:
            locs.append(edata['contact_location'])
    if not locs:
        return None
    return tuple(np.mean(locs, axis=0))


def _node_positions(G):
    """
    Return dict {node: np.array([x,y,z])} for all nodes that have positions.
    Wall nodes are positioned via adjacent edge contact_locations.
    """
    pos = {}
    for n, d in G.nodes(data=True):
        if 'position' in d:
            pos[n] = np.asarray(d['position'], dtype=float)
        elif d.get('is_wall', False):
            p = _wall_position(G, n)
            if p is not None:
                pos[n] = np.asarray(p, dtype=float)
    return pos


def _collect_scalar(G, nodes_or_edges, attr, is_edge=False):
    """Extract scalar attribute values; returns np.ndarray (NaN where missing)."""
    vals = []
    if is_edge:
        for u, v in nodes_or_edges:
            d = G[u][v]
            v_ = d.get(attr, np.nan)
            vals.append(float(v_) if not isinstance(v_, (bool, np.bool_)) else float(v_))
    else:
        for n in nodes_or_edges:
            d = G.nodes[n]
            v_ = d.get(attr, np.nan)
            vals.append(float(v_) if not isinstance(v_, (bool, np.bool_)) else float(v_))
    return np.array(vals, dtype=float)


def _map_scalar_to_range(vals, out_min, out_max, fallback, vmin=None, vmax=None):
    """
    Linearly map scalar values to [out_min, out_max].

    NaN values and degenerate ranges fall back to *fallback*.
    """
    out = np.full(len(vals), float(fallback), dtype=float)
    if len(vals) == 0:
        return out

    valid_mask = ~np.isnan(vals)
    if not valid_mask.any():
        return out

    data = vals[valid_mask]
    lo = float(vmin) if vmin is not None else float(data.min())
    hi = float(vmax) if vmax is not None else float(data.max())
    if hi <= lo:
        return out

    t = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    out[valid_mask] = float(out_min) + (float(out_max) - float(out_min)) * t
    return out


def _make_cube_faces(center, half_size=0.5):
    """Return the 6 face vertex arrays for a cube centred at *center*."""
    cx, cy, cz = center
    h = half_size
    vertices = np.array([
        [cx - h, cy - h, cz - h],
        [cx + h, cy - h, cz - h],
        [cx + h, cy + h, cz - h],
        [cx - h, cy + h, cz - h],
        [cx - h, cy - h, cz + h],
        [cx + h, cy - h, cz + h],
        [cx + h, cy + h, cz + h],
        [cx - h, cy + h, cz + h],
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]
    return faces


def _make_sphere_faces(center, radius, n=8):
    """Return a list of 4-vertex quad faces approximating a sphere."""
    u  = np.linspace(0, 2 * np.pi, n + 1)
    v  = np.linspace(0, np.pi,     n // 2 + 1)
    cx, cy, cz = center
    faces = []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            p00 = [cx + radius * np.cos(u[i])   * np.sin(v[j]),
                   cy + radius * np.sin(u[i])   * np.sin(v[j]),
                   cz + radius * np.cos(v[j])]
            p10 = [cx + radius * np.cos(u[i+1]) * np.sin(v[j]),
                   cy + radius * np.sin(u[i+1]) * np.sin(v[j]),
                   cz + radius * np.cos(v[j])]
            p11 = [cx + radius * np.cos(u[i+1]) * np.sin(v[j+1]),
                   cy + radius * np.sin(u[i+1]) * np.sin(v[j+1]),
                   cz + radius * np.cos(v[j+1])]
            p01 = [cx + radius * np.cos(u[i])   * np.sin(v[j+1]),
                   cy + radius * np.sin(u[i])   * np.sin(v[j+1]),
                   cz + radius * np.cos(v[j+1])]
            faces.append([p00, p10, p11, p01])
    return faces


def _compute_node_alphas(pos, particle_nodes, wall_nodes,
                          alpha_mode='radial',
                          alpha_max=0.95, alpha_min=0.15,
                          alpha_uniform=0.85,
                          G=None, alpha_attr=None, alpha_attr_scale=-1.0):
    """
    Compute per-node alpha (opacity) values.

    Parameters
    ----------
    alpha_mode : 'radial' | 'attr' | 'none'
        'radial'  – alpha is a **continuous linear ramp** from the cylinder
                    axis (r=0) outward to r_max, the outermost particle:
                        alpha(r) = alpha_max + (alpha_min - alpha_max) * r/r_max
                    The cylinder axis is estimated as the mean x-y centroid
                    of all particle positions.
        'attr'    – alpha is mapped linearly from a node attribute:
                        scaled_val = node[alpha_attr] * alpha_attr_scale
                        alpha = alpha_min + (alpha_max - alpha_min) * normalise(scaled_val)
                    Larger scaled value → alpha_max (most opaque).
                    Use alpha_attr='stress_hydro', alpha_attr_scale=-1 so that
                    highly compressive (negative) particles are most opaque.
        'none'    – every node gets the same uniform opacity alpha_uniform.
    alpha_max      : opacity at high-value end (attr mode) or centre (radial)
    alpha_min      : opacity at low-value end (attr mode) or rim (radial)
    alpha_uniform  : opacity when alpha_mode='none'
    G              : NetworkX graph (required for alpha_mode='attr')
    alpha_attr     : str – node attribute name to drive opacity
    alpha_attr_scale : float – multiplier applied to the attribute before
                       normalising (default -1 so compressive stress → large)

    Returns
    -------
    dict {node: float}
    """
    node_alphas = {}

    if alpha_mode == 'none':
        for n in list(particle_nodes) + list(wall_nodes):
            node_alphas[n] = alpha_uniform
        return node_alphas

    if alpha_mode == 'attr' and G is not None and alpha_attr is not None:
        # Collect scaled attribute values for all particle nodes
        raw = np.array([
            float(G.nodes[n].get(alpha_attr, np.nan)) * alpha_attr_scale
            for n in particle_nodes
        ])
        valid_mask = ~np.isnan(raw)
        if valid_mask.any():
            vmin_a = raw[valid_mask].min()
            vmax_a = raw[valid_mask].max()
            span_a = vmax_a - vmin_a if vmax_a > vmin_a else 1e-9
        else:
            vmin_a, span_a = 0.0, 1e-9

        for i, n in enumerate(particle_nodes):
            if valid_mask[i]:
                t = np.clip((raw[i] - vmin_a) / span_a, 0.0, 1.0)
                node_alphas[n] = float(alpha_min + (alpha_max - alpha_min) * t)
            else:
                node_alphas[n] = float(alpha_uniform)

        for n in wall_nodes:
            node_alphas[n] = float(alpha_uniform)
        return node_alphas

    # Estimate cylinder-axis centre from particle x-y positions
    pxy = np.array([[pos[n][0], pos[n][1]] for n in particle_nodes if n in pos])
    if len(pxy) > 0:
        cx, cy = pxy[:, 0].mean(), pxy[:, 1].mean()
        radii = np.sqrt((pxy[:, 0] - cx) ** 2 + (pxy[:, 1] - cy) ** 2)
        r_max = radii.max() if radii.max() > 0 else 1.0
    else:
        cx, cy, r_max = 0.0, 0.0, 1.0

    def _r_alpha(xy):
        r = np.sqrt((xy[0] - cx) ** 2 + (xy[1] - cy) ** 2)
        r_norm = min(r / r_max, 1.0)
        # linear ramp: alpha_max at centre → alpha_min at rim
        return float(alpha_max + (alpha_min - alpha_max) * r_norm)

    for n in particle_nodes:
        node_alphas[n] = _r_alpha(pos[n][:2]) if n in pos else alpha_max

    for n in wall_nodes:
        node_alphas[n] = _r_alpha(pos[n][:2]) if n in pos else 0.65

    return node_alphas


# ---------------------------------------------------------------------------
# Per-graph renderer (one Axes3D)
# ---------------------------------------------------------------------------

def _draw_single_graph(
    ax, G,
    graph_view='core',
    node_attr=None, edge_attr=None,
    node_cmap_name='parula',
    edge_cmap_name='parula',
    node_vmin=None, node_vmax=None,
    edge_vmin=None, edge_vmax=None,
    node_default_color='0.5',
    edge_default_color='0.5',
    elev=30, azim=45,
    cube_half_size=None,
    alpha_mode='radial',
    alpha_max=0.95,
    alpha_min=0.15,
    alpha_uniform=0.85,
    alpha_attr=None,
    alpha_attr_scale=-1.0,
    grid=False,
    node_attr_scale=1.0,
    edge_alpha=None,
    sphere_nodes=False,
    node_resolution=8,
    sphere_scale=1.0,
    marker_size=60,
    node_size_attr=None,
    node_size_attr_scale=1.0,
    node_size_min=20,
    node_size_max=120,
    edge_width=0.7,
    edge_width_attr=None,
    edge_width_attr_scale=1.0,
    edge_width_min=0.2,
    edge_width_max=2.0,
):
    """
    Draw a single NetworkX graph into *ax* (an Axes3D instance).

    Parameters
    ----------
    ax            : Axes3D
    G             : NetworkX Graph
    graph_view    : 'core' or 'full'
    node_attr     : str or None – node attribute used for colour mapping
    edge_attr     : str or None – edge attribute used for colour mapping
    node_cmap_name: str – node colormap name (default 'parula')
    edge_cmap_name: str – edge colormap name (default 'parula')
    node_vmin/max : float or None – colormap range for nodes
    edge_vmin/max : float or None – colormap range for edges
    node_default_color : Matplotlib color spec used when node_attr is None.
    edge_default_color : Matplotlib color spec used when edge_attr is None.
    elev / azim   : viewing angles
    cube_half_size: size of cubes drawn for wall nodes (auto if None)
    alpha_mode    : 'radial' | 'none'
        'radial' – continuous linear ramp:
                   alpha(r) = alpha_max + (alpha_min - alpha_max) * r/r_max
        'none'   – uniform opacity alpha_uniform for all nodes.
    alpha_max       : opacity at cylinder axis (r=0)          (default 0.95)
    alpha_min       : opacity at outermost particle (r=r_max)  (default 0.15)
    alpha_uniform   : opacity when alpha_mode='none'            (default 0.85)
    grid            : bool – show axis grid (default False)
    node_attr_scale : float – multiply node attribute values before colouring
                      (e.g. -1 to plot the negative).  Default 1.0.
    edge_alpha      : float | None – fixed opacity for all edges.  When None
                      (default), each edge uses the mean of its two endpoint
                      node opacities.
    sphere_nodes    : bool – render particles as 3-D sphere meshes instead of
                      scatter markers (default False).  Slower but looks better
                      at high DPI / for publication.
    node_resolution : int   – number of longitude segments for each sphere
                      (default 8, try 12-16 for publication quality).
    sphere_scale    : float – multiplier applied to the sphere radius
                      (default 1.0).  Increase >1 to make spheres bigger,
                      decrease <1 to shrink them.  The base radius is the
                      per-node ``radius`` attribute when present, otherwise
                      half the median contact-edge length.
    marker_size     : float – scatter marker size in points² (default 60).
                      Ignored when sphere_nodes=True.
    node_size_attr  : str | None – node attribute used to scale scatter size
                      (independent of node_attr colour mapping).
    node_size_attr_scale : float – multiplier applied to node_size_attr.
    node_size_min/max : float – output range for marker areas in points².
    edge_width      : float – line width for edges (default 0.7).
    edge_width_attr : str | None – edge attribute used to scale edge widths
                      (independent of edge_attr colour mapping).
    edge_width_attr_scale : float – multiplier on edge_width_attr values.
    edge_width_min/max : float – output range for edge line widths.
    alpha_attr      : str | None – node attribute that drives opacity when
                      alpha_mode='attr'.  E.g. 'stress_hydro'.
    alpha_attr_scale: float – multiplier on alpha_attr before normalising
                      (default -1 so compressive/negative stress → opaque).
    """
    node_cmap = _get_cmap(node_cmap_name)
    edge_cmap = _get_cmap(edge_cmap_name)
    node_default_rgb = np.array(mcolors.to_rgb(node_default_color), dtype=float)
    edge_default_rgb = np.array(mcolors.to_rgb(edge_default_color), dtype=float)
    pos = _node_positions(G)
    if not pos:
        ax.set_title("No positions found")
        return

    all_pos = np.array(list(pos.values()))
    span = np.ptp(all_pos, axis=0)
    typical_size = float(np.median(span[span > 0])) if np.any(span > 0) else 1.0

    if cube_half_size is None:
        cube_half_size = typical_size * 0.03

    # ------------------------------------------------------------------
    # Node / edge lists
    # ------------------------------------------------------------------
    particle_nodes    = [n for n in G.nodes() if not G.nodes[n].get('is_wall', False) and n in pos]
    wall_nodes        = [n for n in G.nodes() if     G.nodes[n].get('is_wall', False) and n in pos]
    edges_positioned  = [(u, v) for u, v in G.edges() if u in pos and v in pos]

    # ------------------------------------------------------------------
    # Per-node alpha
    # ------------------------------------------------------------------
    node_alphas = _compute_node_alphas(
        pos, particle_nodes, wall_nodes,
        alpha_mode=alpha_mode,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        alpha_uniform=alpha_uniform,
        G=G,
        alpha_attr=alpha_attr,
        alpha_attr_scale=alpha_attr_scale,
    )

    # ------------------------------------------------------------------
    # Colour mappings (RGB only; alpha injected from node_alphas)
    # ------------------------------------------------------------------
    # Particle nodes
    if node_attr is not None:
        node_vals = _collect_scalar(G, particle_nodes, node_attr, is_edge=False)
        node_vals = node_vals * node_attr_scale
        valid = node_vals[~np.isnan(node_vals)]
        nmin = node_vmin if node_vmin is not None else (float(valid.min()) if len(valid) else 0.0)
        nmax = node_vmax if node_vmax is not None else (float(valid.max()) if len(valid) else 1.0)
        if nmax == nmin:
            nmax = nmin + 1e-9
        norm_n = mcolors.Normalize(vmin=nmin, vmax=nmax)
        node_rgba = node_cmap(norm_n(node_vals)).copy()          # (N, 4)
        for i, n in enumerate(particle_nodes):
            node_rgba[i, 3] = node_alphas.get(n, alpha_uniform)
    else:
        # Default colour, per-node alpha
        node_rgba = np.array(
            [[node_default_rgb[0], node_default_rgb[1], node_default_rgb[2],
              node_alphas.get(n, alpha_uniform)] for n in particle_nodes],
            dtype=float,
        ) if particle_nodes else np.empty((0, 4))

    # Edge colours (RGB from edge_attr or gray; alpha = mean of endpoint node_alphas)
    if edge_attr is not None:
        edge_vals = _collect_scalar(G, edges_positioned, edge_attr, is_edge=True)
        valid_e = edge_vals[~np.isnan(edge_vals)]
        emin = edge_vmin if edge_vmin is not None else (float(valid_e.min()) if len(valid_e) else 0.0)
        emax = edge_vmax if edge_vmax is not None else (float(valid_e.max()) if len(valid_e) else 1.0)
        if emax == emin:
            emax = emin + 1e-9
        norm_e = mcolors.Normalize(vmin=emin, vmax=emax)
        edge_rgb = edge_cmap(norm_e(edge_vals))[:, :3]           # (M, 3)
    else:
        edge_rgb = np.tile(edge_default_rgb, (len(edges_positioned), 1))

    # Node sizes (scatter mode)
    if node_size_attr is not None and particle_nodes:
        size_vals = _collect_scalar(G, particle_nodes, node_size_attr, is_edge=False)
        size_vals = size_vals * float(node_size_attr_scale)
        node_sizes = _map_scalar_to_range(
            size_vals,
            out_min=node_size_min,
            out_max=node_size_max,
            fallback=marker_size,
        )
    else:
        node_sizes = np.full(len(particle_nodes), float(marker_size), dtype=float)

    # Edge widths
    if edge_width_attr is not None and edges_positioned:
        width_vals = _collect_scalar(G, edges_positioned, edge_width_attr, is_edge=True)
        width_vals = width_vals * float(edge_width_attr_scale)
        edge_widths = _map_scalar_to_range(
            width_vals,
            out_min=edge_width_min,
            out_max=edge_width_max,
            fallback=edge_width,
        )
    else:
        edge_widths = np.full(len(edges_positioned), float(edge_width), dtype=float)

    # ------------------------------------------------------------------
    # Draw edges
    # ------------------------------------------------------------------
    edge_segs   = []
    edge_colors = []
    for i, (u, v) in enumerate(edges_positioned):
        seg = [pos[u], pos[v]]
        if edge_alpha is not None:
            a = float(edge_alpha)
        else:
            a = 0.5 * (node_alphas.get(u, alpha_uniform) + node_alphas.get(v, alpha_uniform))
        r, g_c, b = edge_rgb[i]
        edge_segs.append(seg)
        edge_colors.append((r, g_c, b, float(a)))

    if edge_segs:
        lc = Line3DCollection(edge_segs, colors=edge_colors, linewidths=edge_widths, zorder=1)
        ax.add_collection3d(lc)

    # ------------------------------------------------------------------
    # Draw particle nodes
    # ------------------------------------------------------------------
    if particle_nodes and len(node_rgba) > 0:
        if sphere_nodes:
            # ------------------------------------------------------------------
            # Sphere radius: per-node 'radius' attribute → fallback to half the
            # median contact-edge length (particles are nearly touching so
            # center-to-center ≈ 2*radius).
            # ------------------------------------------------------------------
            node_radii = {}
            for n in particle_nodes:
                r_attr = G.nodes[n].get('radius', None)
                node_radii[n] = float(r_attr) if r_attr is not None else None

            if all(v is None for v in node_radii.values()):
                # Fallback: half of median edge length among positioned edges
                edge_lengths = [
                    float(np.linalg.norm(pos[u] - pos[v]))
                    for u, v in edges_positioned
                    if not G.nodes[u].get('is_wall', False)
                    and not G.nodes[v].get('is_wall', False)
                ]
                if edge_lengths:
                    fallback_r = float(np.median(edge_lengths)) * 0.5
                else:
                    fallback_r = typical_size * 0.025
                for n in particle_nodes:
                    node_radii[n] = fallback_r

            # One Poly3DCollection per sphere: only scalar set_alpha() is
            # reliable in matplotlib 3D — per-face RGBA alpha is ignored.
            for i, n in enumerate(particle_nodes):
                sphere_r = node_radii[n] * sphere_scale
                faces = _make_sphere_faces(pos[n], sphere_r, n=node_resolution)
                rgb   = node_rgba[i, :3]        # (R, G, B)
                alpha = float(node_rgba[i, 3])  # scalar alpha for this node
                face_colors = np.tile(rgb, (len(faces), 1))
                poly = Poly3DCollection(
                    faces,
                    facecolors=face_colors,
                    edgecolors='none',
                    zorder=3,
                )
                poly.set_alpha(alpha)
                ax.add_collection3d(poly)
        else:
            ppos = np.array([pos[n] for n in particle_nodes])
            ax.scatter(
                ppos[:, 0], ppos[:, 1], ppos[:, 2],
                c=node_rgba, s=node_sizes,
                depthshade=True, zorder=3, linewidths=0,
            )

    # ------------------------------------------------------------------
    # Draw wall nodes (cubes via Poly3DCollection)
    # ------------------------------------------------------------------
    if wall_nodes and graph_view == 'full':
        # Colour for wall nodes
        if node_attr is not None:
            wall_vals = _collect_scalar(G, wall_nodes, node_attr, is_edge=False)
            wall_rgb  = node_cmap(norm_n(wall_vals))[:, :3]
        else:
            wall_rgb = np.tile(node_default_rgb, (len(wall_nodes), 1))

        all_cube_faces  = []
        all_cube_colors = []
        for i, wn in enumerate(wall_nodes):
            faces = _make_cube_faces(pos[wn], half_size=cube_half_size)
            a     = node_alphas.get(wn, 0.65)
            clr   = list(wall_rgb[i]) + [float(a)]
            all_cube_faces.extend(faces)
            all_cube_colors.extend([clr] * len(faces))

        if all_cube_faces:
            poly = Poly3DCollection(
                all_cube_faces,
                facecolors=all_cube_colors,
                edgecolors='none',
                zorder=4,
            )
            ax.add_collection3d(poly)

    # ------------------------------------------------------------------
    # Axis limits, labels, white panes, grid
    # ------------------------------------------------------------------
    mn = all_pos.min(axis=0)
    mx = all_pos.max(axis=0)
    pad = typical_size * 0.05

    # Equal-aspect: use the same half-span for all axes so spheres aren't distorted
    center    = 0.5 * (mn + mx)
    half_span = 0.5 * (mx - mn) + pad
    max_half  = half_span.max()

    ax.set_xlim(center[0] - max_half, center[0] + max_half)
    ax.set_ylim(center[1] - max_half, center[1] + max_half)
    ax.set_zlim(center[2] - max_half, center[2] + max_half)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X', fontsize=8, labelpad=3)
    ax.set_ylabel('Y', fontsize=8, labelpad=3)
    ax.set_zlabel('Z', fontsize=8, labelpad=3)
    ax.tick_params(labelsize=6)

    # Hide box: white panes (blend into background) with no visible border
    for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis_obj.pane.fill = True
        axis_obj.pane.set_facecolor('white')
        axis_obj.pane.set_edgecolor('none')

    # Grid control
    if not grid:
        ax.grid(False)
        for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis_obj._axinfo['grid']['color'] = (1, 1, 1, 0)

    # Title
    angle_label = G.graph.get('angle_label', '?')
    sim_idx     = G.graph.get('sim_idx', '?')
    view_tag    = 'full' if graph_view == 'full' else 'core'
    ax.set_title(f"{angle_label}  sim={sim_idx}  ({view_tag})", fontsize=9, pad=4)

    ax.view_init(elev=elev, azim=azim)
    ax.invert_zaxis()   # flip Z so high-Z end appears at bottom (loading direction)


# ---------------------------------------------------------------------------
# Auto-filename helper
# ---------------------------------------------------------------------------

def _make_auto_filename(geometries, sim_indices, graph_view, node_attr, edge_attr, bg, fmt):
    """
    Build a descriptive filename from plot parameters.

    Example output::

        graph3d_core_0deg-15deg_s3-s7_n-stress_vm_e-normal_force_white.png
    """
    geom_str = '-'.join(geometries) if geometries else 'allgeom'
    sim_str  = '-'.join(f's{i}' for i in sim_indices) if sim_indices else 'srnd'
    n_str    = f'n-{node_attr}' if node_attr else 'n-gray'
    e_str    = f'e-{edge_attr}' if edge_attr else 'e-gray'
    parts    = [geom_str, sim_str, n_str, e_str, graph_view, bg]
    name     = 'graph3d_' + '_'.join(p.replace(' ', '_').replace('/', '_') for p in parts)
    return f'{name}.{fmt.lstrip(".")}'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_graph_3d(
    graph_dict,
    geometries=None,
    sim_indices=None,
    graph_view='core',
    node_attr=None,
    edge_attr=None,
    cmap='parula',
    node_cmap=None,
    edge_cmap=None,
    node_vmin=None,
    node_vmax=None,
    edge_vmin=None,
    edge_vmax=None,
    show_node_colorbar=True,
    show_edge_colorbar=False,
    node_default_color='0.5',
    edge_default_color='0.5',
    elev=30,
    azim=45,
    bg='white',
    figsize=None,
    separate_figures=True,
    grid=False,
    alpha_mode='radial',
    alpha_max=0.95,
    alpha_min=0.15,
    alpha_uniform=0.85,
    node_attr_scale=1.0,
    alpha_attr=None,
    alpha_attr_scale=-1.0,
    edge_alpha=None,
    sphere_nodes=False,
    node_resolution=8,
    sphere_scale=1.0,
    marker_size=60,
    node_size_attr=None,
    node_size_attr_scale=1.0,
    node_size_min=20,
    node_size_max=120,
    edge_width=0.7,
    edge_width_attr=None,
    edge_width_attr_scale=1.0,
    edge_width_min=0.2,
    edge_width_max=2.0,
    save_path=None,
    save_dir=None,
    fmt='png',
    dpi=150,
    show=True,
):
    """
    3-D plot of one or more graphs from *graph_dict*.

    Parameters
    ----------
    graph_dict  : dict
        Loaded ``graph_dict_labeled.pkl``.  Keys are angle labels
        (``'0deg'``, ``'15deg'``, ``'30deg'``, ``'45deg'``).
        Values are ``{'full': [...], 'core': [...]}``.
    geometries  : list[str] | None
        Which angle labels to plot.  ``None`` → all four.
    sim_indices : list[int] | int | None
        Simulation index per geometry.  ``None`` → random.
    graph_view  : ``'core'`` | ``'full'``
    node_attr   : str | None   – node attribute for colour; ``None`` → gray.
    edge_attr   : str | None   – edge attribute for colour; ``None`` → gray.
    cmap        : str          – colormap (default ``'parula'``).
    node_cmap   : str | Colormap | None – node colormap. ``None`` → use *cmap*.
    edge_cmap   : str | Colormap | None – edge colormap. ``None`` → use *cmap*.
    node_vmin / node_vmax : float | None  – colormap range for nodes.
    edge_vmin / edge_vmax : float | None  – colormap range for edges.
    show_node_colorbar : bool – show node colorbar when node_attr is mapped.
    show_edge_colorbar : bool – show edge colorbar when edge_attr is mapped.
    node_default_color : Matplotlib color spec used when node_attr is None.
    edge_default_color : Matplotlib color spec used when edge_attr is None.
    elev / azim : float        – 3-D viewing angle.
    bg          : ``'white'`` | ``'transparent'``.
    figsize     : (float, float) | None  – per-figure size in inches.
    separate_figures : bool
        ``True`` (default) – one figure per graph, returned as a list.
        ``False``           – all graphs in one figure with subplots.
    grid        : bool  – show axis grid (default ``False``).
    alpha_mode  : ``'radial'`` | ``'none'``
        ``'radial'`` – opacity decreases with 2-D radial distance from the
                       estimated cylinder axis (x-y mean of all particles).
                       Core → ``alpha_center``; outermost ring → ``alpha_boundary``.
        ``'none'``   – uniform opacity ``alpha_uniform`` for every node.
    alpha_max        : float  – opacity at r = 0  (default 0.95).
    alpha_min        : float  – opacity at r_max  (default 0.15).
    node_attr_scale  : float  – multiply node values before colouring; use -1 to
                       plot the negative of the attribute (default 1.0).
    edge_alpha       : float | None – fixed opacity for all edges.  ``None``
                       (default) → each edge uses the mean of its two endpoint
                       node opacities.
    sphere_nodes     : bool  – render particles as 3-D sphere meshes instead
                       of scatter markers (default False).
    node_resolution  : int   – sphere longitude segments (default 8).
    sphere_scale     : float – radius multiplier for sphere meshes (default 1.0).
                       Uses per-node ``radius`` attribute when available,
                       else half the median edge length.
    marker_size      : float – scatter marker size in points² (default 60).
                       Ignored when sphere_nodes=True.
    node_size_attr   : str | None – node attribute to scale marker sizes.
    node_size_attr_scale : float – multiplier applied before size mapping.
    node_size_min/max: float – marker size mapping range in points².
    edge_width       : float – line width for edges (default 0.7).
    edge_width_attr  : str | None – edge attribute to scale line width.
    edge_width_attr_scale : float – multiplier applied before width mapping.
    edge_width_min/max : float – edge width mapping range.
    alpha_uniform  : float  – opacity when ``alpha_mode='none'`` (default 0.85).
    save_path   : str | None  – full path; takes precedence over *save_dir*.
    save_dir    : str | None  – directory; filename(s) are auto-generated.
        When ``separate_figures=True``, each figure is saved as::

            graph3d_<view>_<geom>_s<sim>_n-<node_attr>_e-<edge_attr>_<bg>.<fmt>
    fmt         : str   – format for *save_dir* saves (default ``'png'``).
    dpi         : int   – resolution for saved figures.
    show        : bool  – call ``plt.show()`` (default ``True``).

    Returns
    -------
    list[Figure] when ``separate_figures=True``, else Figure.

    Examples
    --------
    >>> from graph3d_viz import plot_graph_3d
    >>> plot_graph_3d(GL_final)                          # random, gray, radial alpha
    >>> plot_graph_3d(GL_final, node_attr='stress_hydro',
    ...               save_dir='/my/figs')               # one PNG per geometry
    >>> plot_graph_3d(GL_final, alpha_mode='none',
    ...               separate_figures=False)            # all in one figure
    """
    # ── Resolve geometries & sim_indices ────────────────────────────────
    available = list(graph_dict.keys())
    if geometries is None:
        geometries = available
    else:
        # Validate
        for g in geometries:
            if g not in graph_dict:
                raise ValueError(f"Geometry '{g}' not found. Available: {available}")

    n_plots = len(geometries)
    n_sims  = {g: len(graph_dict[g][graph_view]) for g in geometries}

    if sim_indices is None:
        sim_indices = [random.randrange(n_sims[g]) for g in geometries]
    elif isinstance(sim_indices, int):
        sim_indices = [sim_indices] * n_plots
    else:
        if len(sim_indices) != n_plots:
            raise ValueError(
                f"Length of sim_indices ({len(sim_indices)}) must match "
                f"length of geometries ({n_plots})."
            )

    # ── Collect graphs ───────────────────────────────────────────────────
    graphs = []
    for geom, sidx in zip(geometries, sim_indices):
        g_list = graph_dict[geom][graph_view]
        if sidx >= len(g_list) or sidx < 0:
            raise IndexError(f"sim_index {sidx} out of range for geometry '{geom}' "
                             f"({len(g_list)} simulations).")
        graphs.append(g_list[sidx])

    # ── Compute shared colour ranges for selected graphs ─
    shared_node_vmin = node_vmin
    shared_node_vmax = node_vmax
    shared_edge_vmin = edge_vmin
    shared_edge_vmax = edge_vmax

    if node_attr is not None and (node_vmin is None or node_vmax is None):
        all_nvals = []
        for G in graphs:
            pnodes = [n for n in G.nodes() if not G.nodes[n].get('is_wall', False)]
            vals = _collect_scalar(G, pnodes, node_attr, is_edge=False) * node_attr_scale
            all_nvals.extend(vals[~np.isnan(vals)])
        if all_nvals:
            auto_nmin = float(np.min(all_nvals))
            auto_nmax = float(np.max(all_nvals))
            if shared_node_vmin is None:
                shared_node_vmin = auto_nmin
            if shared_node_vmax is None:
                shared_node_vmax = auto_nmax

    if edge_attr is not None and (edge_vmin is None or edge_vmax is None):
        all_evals = []
        for G in graphs:
            pos = _node_positions(G)
            edges_p = [(u, v) for u, v in G.edges() if u in pos and v in pos]
            vals = _collect_scalar(G, edges_p, edge_attr, is_edge=True)
            all_evals.extend(vals[~np.isnan(vals)])
        if all_evals:
            auto_emin = float(np.min(all_evals))
            auto_emax = float(np.max(all_evals))
            if shared_edge_vmin is None:
                shared_edge_vmin = auto_emin
            if shared_edge_vmax is None:
                shared_edge_vmax = auto_emax

    # ── Helpers ──────────────────────────────────────────────────────────
    transparent = (bg == 'transparent')
    fig_size = figsize if figsize is not None else (6.0, 6.0)

    def _resolve_cmap(cmap_choice, fallback):
        if cmap_choice is None:
            cmap_choice = fallback
        if isinstance(cmap_choice, str):
            return _get_cmap(cmap_choice), cmap_choice
        return cmap_choice, 'custom'

    node_cmap_obj, node_cmap_name_str = _resolve_cmap(node_cmap, cmap)
    edge_cmap_obj, edge_cmap_name_str = _resolve_cmap(edge_cmap, cmap)

    node_color_mapped = (node_attr is not None)
    edge_color_mapped = (edge_attr is not None)
    node_has_variation = (
        shared_node_vmin is not None and
        shared_node_vmax is not None and
        shared_node_vmax > shared_node_vmin
    )
    edge_has_variation = (
        shared_edge_vmin is not None and
        shared_edge_vmax is not None and
        shared_edge_vmax > shared_edge_vmin
    )
    do_node_colorbar = bool(show_node_colorbar and node_color_mapped and node_has_variation)
    do_edge_colorbar = bool(show_edge_colorbar and edge_color_mapped and edge_has_variation)

    def _make_ax(fig_):
        ax_ = fig_.add_subplot(1, 1, 1, projection='3d')
        if transparent:
            ax_.set_facecolor('none')
            ax_.patch.set_alpha(0.0)
        else:
            ax_.set_facecolor('white')
        return ax_

    def _add_colorbars(fig_, ax_):
        if do_node_colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=node_cmap_obj,
                norm=mcolors.Normalize(vmin=shared_node_vmin, vmax=shared_node_vmax),
            )
            sm.set_array([])
            cb = fig_.colorbar(sm, ax=ax_, orientation='vertical',
                               fraction=0.04, pad=0.1, shrink=0.6)
            cb.set_label(node_attr, fontsize=9)

        if do_edge_colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=edge_cmap_obj,
                norm=mcolors.Normalize(vmin=shared_edge_vmin, vmax=shared_edge_vmax),
            )
            sm.set_array([])
            cb = fig_.colorbar(sm, ax=ax_, orientation='vertical',
                               fraction=0.04, pad=0.02 if do_node_colorbar else 0.1, shrink=0.6)
            cb.set_label(edge_attr, fontsize=9)

    def _save_fig(fig_, geom, sidx):
        if save_path is not None:
            rp = save_path
        elif save_dir is not None:
            fname = _make_auto_filename(
                [geom], [sidx], graph_view,
                node_attr, edge_attr, bg, fmt,
            )
            rp = os.path.join(save_dir, fname)
        else:
            return
        os.makedirs(os.path.dirname(os.path.abspath(rp)), exist_ok=True)
        fig_.savefig(rp, dpi=dpi, transparent=transparent, bbox_inches='tight')
        print(f"✅ Saved: {rp}")

    common_draw_kwargs = dict(
        graph_view=graph_view,
        node_attr=node_attr, edge_attr=edge_attr,
        node_cmap_name=node_cmap_name_str,
        edge_cmap_name=edge_cmap_name_str,
        node_vmin=shared_node_vmin, node_vmax=shared_node_vmax,
        edge_vmin=shared_edge_vmin, edge_vmax=shared_edge_vmax,
        node_default_color=node_default_color,
        edge_default_color=edge_default_color,
        elev=elev, azim=azim,
        alpha_mode=alpha_mode,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        alpha_uniform=alpha_uniform,
        grid=grid,
        node_attr_scale=node_attr_scale,
        alpha_attr=alpha_attr,
        alpha_attr_scale=alpha_attr_scale,
        edge_alpha=edge_alpha,
        sphere_nodes=sphere_nodes,
        node_resolution=node_resolution,
        sphere_scale=sphere_scale,
        marker_size=marker_size,
        node_size_attr=node_size_attr,
        node_size_attr_scale=node_size_attr_scale,
        node_size_min=node_size_min,
        node_size_max=node_size_max,
        edge_width=edge_width,
        edge_width_attr=edge_width_attr,
        edge_width_attr_scale=edge_width_attr_scale,
        edge_width_min=edge_width_min,
        edge_width_max=edge_width_max,
    )

    # ── Separate figure per graph (default) ──────────────────────────────
    if separate_figures:
        figs = []
        for G, geom, sidx in zip(graphs, geometries, sim_indices):
            fig_ = plt.figure(figsize=fig_size,
                              facecolor='none' if transparent else 'white')
            ax_  = _make_ax(fig_)
            _draw_single_graph(ax_, G, **common_draw_kwargs)
            _add_colorbars(fig_, ax_)
            fig_.tight_layout()
            _save_fig(fig_, geom, sidx)
            if show:
                plt.show()
            figs.append(fig_)
        return figs

    # ── All graphs in one figure ──────────────────────────────────────────
    if n_plots <= 4:
        ncols, nrows = n_plots, 1
    else:
        ncols = 4
        nrows = int(np.ceil(n_plots / ncols))

    multi_size = figsize if figsize is not None else (5.5 * ncols, 5.5 * nrows)
    fig  = plt.figure(figsize=multi_size,
                      facecolor='none' if transparent else 'white')
    axes = []
    for i, (G, geom, sidx) in enumerate(zip(graphs, geometries, sim_indices)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        if transparent:
            ax.set_facecolor('none')
            ax.patch.set_alpha(0.0)
        else:
            ax.set_facecolor('white')
        _draw_single_graph(ax, G, **common_draw_kwargs)
        axes.append(ax)

    # Shared colourbars for multi-panel figure
    if do_node_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=node_cmap_obj,
            norm=mcolors.Normalize(vmin=shared_node_vmin, vmax=shared_node_vmax),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, orientation='vertical',
                          fraction=0.012, pad=0.04, shrink=0.6)
        cb.set_label(node_attr, fontsize=9)

    if do_edge_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=edge_cmap_obj,
            norm=mcolors.Normalize(vmin=shared_edge_vmin, vmax=shared_edge_vmax),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, orientation='vertical',
                          fraction=0.012, pad=0.01 if do_node_colorbar else 0.04, shrink=0.6)
        cb.set_label(edge_attr, fontsize=9)

    fig.tight_layout()

    # For multi-panel, save_path saves the single figure; save_dir uses all-geom name
    if save_path is not None:
        rp = save_path
    elif save_dir is not None:
        fname = _make_auto_filename(geometries, sim_indices, graph_view,
                                    node_attr, edge_attr, bg, fmt)
        rp = os.path.join(save_dir, fname)
    else:
        rp = None
    if rp is not None:
        os.makedirs(os.path.dirname(os.path.abspath(rp)), exist_ok=True)
        fig.savefig(rp, dpi=dpi, transparent=transparent, bbox_inches='tight')
        print(f"✅ Saved: {rp}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Convenience: list available attributes for exploration
# ---------------------------------------------------------------------------

def inspect_graph(graph_dict, geometry=None, sim_idx=0, graph_view='core'):
    """
    Print all node and edge attributes of a selected graph.

    Parameters
    ----------
    graph_dict : dict
    geometry   : str | None  – angle label; None → first available
    sim_idx    : int         – simulation index (default 0)
    graph_view : 'core' | 'full'
    """
    if geometry is None:
        geometry = list(graph_dict.keys())[0]
    G = graph_dict[geometry][graph_view][sim_idx]

    print(f"\n{'='*60}")
    print(f"  Graph: {geometry} | sim_idx={sim_idx} | view={graph_view}")
    print(f"  Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}")
    print(f"{'='*60}")

    # Sample node attrs
    particle_nodes = [n for n in G.nodes() if not G.nodes[n].get('is_wall', False)]
    wall_nodes     = [n for n in G.nodes() if     G.nodes[n].get('is_wall', False)]

    if particle_nodes:
        sample = particle_nodes[0]
        print(f"\n▶  Particle node attributes (sample node {sample!r}):")
        for k, v in G.nodes[sample].items():
            print(f"     {k:30s} : {type(v).__name__:12s}  = {v!r:.60s}")

    if wall_nodes:
        sample = wall_nodes[0]
        print(f"\n▶  Wall node attributes (sample node {sample!r}):")
        for k, v in G.nodes[sample].items():
            print(f"     {k:30s} : {type(v).__name__:12s}  = {v!r:.60s}")

    if G.number_of_edges() > 0:
        eu, ev = next(iter(G.edges()))
        print(f"\n▶  Edge attributes (sample edge ({eu!r}, {ev!r})):")
        for k, v in G[eu][ev].items():
            print(f"     {k:30s} : {type(v).__name__:12s}  = {v!r:.60s}")

    print()


# ---------------------------------------------------------------------------
# CLI / demo usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='3-D graph visualisation')
    parser.add_argument('pkl', help='Path to graph_dict_labeled.pkl')
    parser.add_argument('--geometries', nargs='+', default=None,
                        help='Angle labels to plot, e.g. 0deg 45deg')
    parser.add_argument('--sim_indices', nargs='*', type=int, default=None,
                        help='Sim indices (one per geometry)')
    parser.add_argument('--view', choices=['core', 'full'], default='core',
                        help='Graph view: core (default) or full')
    parser.add_argument('--node_attr', default=None,
                        help='Node attribute for colour, e.g. stress_vm')
    parser.add_argument('--edge_attr', default=None,
                        help='Edge attribute for colour, e.g. normal_force')
    parser.add_argument('--cmap',   default='parula',
                        help='Colormap name (default: parula)')
    parser.add_argument('--node_cmap', default=None,
                        help='Node colormap (default: uses --cmap)')
    parser.add_argument('--edge_cmap', default=None,
                        help='Edge colormap (default: uses --cmap)')
    parser.add_argument('--node_vmin', type=float, default=None)
    parser.add_argument('--node_vmax', type=float, default=None)
    parser.add_argument('--edge_vmin', type=float, default=None)
    parser.add_argument('--edge_vmax', type=float, default=None)
    parser.add_argument('--hide_node_colorbar', action='store_true',
                        help='Hide node colorbar')
    parser.add_argument('--show_edge_colorbar', action='store_true',
                        help='Show edge colorbar')
    parser.add_argument('--node_default_color', default='0.5',
                        help='Default node color when node_attr is not set')
    parser.add_argument('--edge_default_color', default='0.5',
                        help='Default edge color when edge_attr is not set')
    parser.add_argument('--elev',  type=float, default=30,
                        help='Elevation angle (default 30)')
    parser.add_argument('--azim',  type=float, default=45,
                        help='Azimuth angle (default 45)')
    parser.add_argument('--bg',             choices=['white', 'transparent'], default='white')
    parser.add_argument('--no_separate',    action='store_true',
                        help='Put all graphs in one figure instead of separate figures')
    parser.add_argument('--grid',           action='store_true',
                        help='Show axis grid (default: hidden)')
    parser.add_argument('--alpha_mode',     choices=['radial', 'none'], default='radial',
                        help='Transparency mode (default: radial)')
    parser.add_argument('--alpha_max',       type=float, default=0.95,
                        help='Opacity at cylinder centre (default 0.95)')
    parser.add_argument('--alpha_min',       type=float, default=0.15,
                        help='Opacity at outermost radius (default 0.15)')
    parser.add_argument('--alpha_uniform',  type=float, default=0.85,
                        help='Uniform opacity when alpha_mode=none (default 0.85)')
    parser.add_argument('--node_size_attr', default=None,
                        help='Node attribute to scale marker size')
    parser.add_argument('--node_size_scale', type=float, default=1.0,
                        help='Multiplier for node size attribute')
    parser.add_argument('--node_size_min', type=float, default=20.0,
                        help='Minimum node marker size (points^2)')
    parser.add_argument('--node_size_max', type=float, default=120.0,
                        help='Maximum node marker size (points^2)')
    parser.add_argument('--edge_width_attr', default=None,
                        help='Edge attribute to scale line width')
    parser.add_argument('--edge_width_scale', type=float, default=1.0,
                        help='Multiplier for edge width attribute')
    parser.add_argument('--edge_width_min', type=float, default=0.2,
                        help='Minimum edge width')
    parser.add_argument('--edge_width_max', type=float, default=2.0,
                        help='Maximum edge width')
    parser.add_argument('--save',           default=None,
                        help='Full save path including filename')
    parser.add_argument('--save_dir',       default=None,
                        help='Directory to save; filename is auto-generated')
    parser.add_argument('--fmt',            default='png',
                        help='File format when using --save_dir (default: png)')
    parser.add_argument('--dpi',            type=int, default=150)
    parser.add_argument('--inspect', action='store_true',
                        help='Print attribute names and exit')
    args = parser.parse_args()

    print(f'Loading {args.pkl} …')
    with open(args.pkl, 'rb') as fh:
        GL = pickle.load(fh)
    print(f'Keys: {list(GL.keys())}')

    if args.inspect:
        inspect_graph(GL, geometry=args.geometries[0] if args.geometries else None,
                      graph_view=args.view)
    else:
        plot_graph_3d(
            GL,
            geometries=args.geometries,
            sim_indices=args.sim_indices,
            graph_view=args.view,
            node_attr=args.node_attr,
            edge_attr=args.edge_attr,
            cmap=args.cmap,
            node_cmap=args.node_cmap,
            edge_cmap=args.edge_cmap,
            node_vmin=args.node_vmin, node_vmax=args.node_vmax,
            edge_vmin=args.edge_vmin, edge_vmax=args.edge_vmax,
            show_node_colorbar=not args.hide_node_colorbar,
            show_edge_colorbar=args.show_edge_colorbar,
            node_default_color=args.node_default_color,
            edge_default_color=args.edge_default_color,
            elev=args.elev, azim=args.azim,
            bg=args.bg,
            separate_figures=not args.no_separate,
            grid=args.grid,
            alpha_mode=args.alpha_mode,
            alpha_max=args.alpha_max,
            alpha_min=args.alpha_min,
            alpha_uniform=args.alpha_uniform,
            node_size_attr=args.node_size_attr,
            node_size_attr_scale=args.node_size_scale,
            node_size_min=args.node_size_min,
            node_size_max=args.node_size_max,
            edge_width_attr=args.edge_width_attr,
            edge_width_attr_scale=args.edge_width_scale,
            edge_width_min=args.edge_width_min,
            edge_width_max=args.edge_width_max,
            save_path=args.save,
            save_dir=args.save_dir,
            fmt=args.fmt,
            dpi=args.dpi,
        )
