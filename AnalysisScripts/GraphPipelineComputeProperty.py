import argparse
import os
import pandas as pd

from GraphPipelineCommon import (
    PAIR_EDGE_DIR,
    PAIR_EDGE_EXPORT_CHUNK_SIZE,
    PAIR_EDGE_EXPORT_N_JOBS,
    NODE_CONN_N_JOBS,
    NODE_CONN_VERBOSE,
    compute_curvature_patch,
    compute_loop_patch,
    compute_nfd_patch,
    compute_node_connectivity_patch,
    compute_pair_edge_patch,
    compute_topology_spectral_patch,
    _load_pickle,
    _patch_path,
    _raw_graph_path,
    _save_pickle,
    PIPELINE_OUT_PATH,
)


GROUPS = {'topology', 'loop', 'pair_edge', 'node_connectivity', 'curvature', 'nfd'}


def _load_slice_index():
    slices_path = os.path.join(PIPELINE_OUT_PATH, 'simulation_slices.csv')
    if not os.path.exists(slices_path):
        raise FileNotFoundError(f'Missing slice index: {slices_path}. Run GraphPipelineBuildGraphs.py first.')
    return pd.read_csv(slices_path)


def _selected_rows(df, geometry=None, sim_idx=None):
    rows = df
    if geometry is not None:
        rows = rows[rows['label'].astype(str).str.lower() == geometry.lower()]
    if sim_idx is not None:
        rows = rows[rows['sim_idx'].astype(int) == int(sim_idx)]
    return rows.sort_values(['label', 'sim_idx'])


def compute_patch(group, G_full):
    if group == 'topology':
        return compute_topology_spectral_patch(G_full)
    if group == 'loop':
        return compute_loop_patch(G_full)
    if group == 'pair_edge':
        return compute_pair_edge_patch(
            G_full,
            pair_edge_out_dir=PIPELINE_OUT_PATH,
            n_jobs=PAIR_EDGE_EXPORT_N_JOBS,
            chunk_size=PAIR_EDGE_EXPORT_CHUNK_SIZE,
        )
    if group == 'node_connectivity':
        return compute_node_connectivity_patch(
            G_full,
            n_jobs=NODE_CONN_N_JOBS,
            verbose=NODE_CONN_VERBOSE,
        )
    if group == 'curvature':
        return compute_curvature_patch(G_full)
    if group == 'nfd':
        return compute_nfd_patch(G_full)
    raise ValueError(f'Unknown property group: {group}')


def run_property_group(group, geometry=None, sim_idx=None):
    if group not in GROUPS:
        raise ValueError(f'Unknown property group {group!r}; expected one of {sorted(GROUPS)}')

    os.makedirs(PAIR_EDGE_DIR, exist_ok=True)
    slice_df = _load_slice_index()
    rows = _selected_rows(slice_df, geometry=geometry, sim_idx=sim_idx)
    if rows.empty:
        raise ValueError('No raw graph rows matched the requested geometry/simulation filter.')

    print(f"Computing property group '{group}' for {len(rows)} graph(s)")
    for _, row in rows.iterrows():
        label = row['label']
        sim = int(row['sim_idx'])
        raw_path = _raw_graph_path(label, sim)
        print(f"[{label} sim {sim:03d}] loading {raw_path}")
        G_full = _load_pickle(raw_path)
        patch = compute_patch(group, G_full)
        out_path = _patch_path(group, label, sim)
        _save_pickle(patch, out_path)
        print(f"[{label} sim {sim:03d}] saved {group} patch: {out_path}")


def main(default_group=None):
    parser = argparse.ArgumentParser(description='Compute one staged graph-property group.')
    if default_group is None:
        parser.add_argument('--group', required=True, choices=sorted(GROUPS))
    parser.add_argument('--geometry', default=os.environ.get('GRAPHPIPE_GEOMETRY'))
    parser.add_argument('--sim-idx', type=int, default=os.environ.get('GRAPHPIPE_SIM_IDX'))
    args = parser.parse_args()

    group = default_group or args.group
    run_property_group(group, geometry=args.geometry, sim_idx=args.sim_idx)


if __name__ == '__main__':
    main()
