import os
import pickle
import pandas as pd

from GraphPipelineCommon import (
    PIPELINE_OUT_PATH,
    PAIR_EDGE_DIR,
    PLOT_FORCE_DISTRIBUTIONS,
    FORCE_DIST_BINS,
    FORCE_DIST_LOGX,
    FORCE_DIST_COLUMN,
    HIST_LINE_WIDTH,
    THRESHOLD_LINE_WIDTH,
    THRESHOLD_LINE_COLOR,
    USE_GLOBAL_HIGH_FORCE_THRESHOLD,
    USE_CORE_FOR_THRESHOLD,
    REFERENCE_GEOMETRY_LABEL,
    apply_property_patch,
    label_high_force_edges,
    plot_force_distributions_row,
    _build_feature_tables,
    _graph_id_fields,
    _load_pickle,
    _make_core_graph,
    _patch_path,
    _raw_graph_path,
    _serialize_value,
)


REQUIRED_GROUPS = ['topology', 'loop', 'pair_edge', 'node_connectivity', 'curvature', 'nfd']


def _load_slice_index():
    slices_path = os.path.join(PIPELINE_OUT_PATH, 'simulation_slices.csv')
    if not os.path.exists(slices_path):
        raise FileNotFoundError(f'Missing slice index: {slices_path}')
    return pd.read_csv(slices_path)


def _write_simulation_slices_txt(slice_df, out_dir):
    header = [
        'label',
        'sim_idx',
        'start_idx_contact',
        'num_contacts',
        'end_idx_contact',
        'start_idx_particle',
        'num_particles',
        'end_idx_particle',
    ]
    with open(os.path.join(out_dir, 'simulation_slices.txt'), 'w') as handle:
        handle.write('\t'.join(header) + '\n')
        for _, entry in slice_df.iterrows():
            handle.write('\t'.join(str(entry[k]) for k in header) + '\n')


def main():
    os.makedirs(PIPELINE_OUT_PATH, exist_ok=True)
    slice_df = _load_slice_index().sort_values(['label', 'sim_idx'])

    graph_dict = {}
    pair_edge_index_records = []
    angle_order = []

    for label, geom_df in slice_df.groupby('label', sort=False):
        angle_order.append(label)
        graph_list_core = []
        graph_list_full = []

        for _, row in geom_df.iterrows():
            sim_idx = int(row['sim_idx'])
            G_full = _load_pickle(_raw_graph_path(label, sim_idx))
            G_core = _make_core_graph(G_full)

            for group in REQUIRED_GROUPS:
                patch_file = _patch_path(group, label, sim_idx)
                if not os.path.exists(patch_file):
                    raise FileNotFoundError(f'Missing {group} patch for {label} sim {sim_idx:03d}: {patch_file}')
                patch = _load_pickle(patch_file)
                apply_property_patch(G_full, G_core, patch)
                pair_entry = patch.get('extra', {}).get('pair_edge_index_entry')
                if pair_entry is not None:
                    pair_edge_index_records.append(pair_entry)

            for node in G_core.nodes:
                if node in G_full.nodes:
                    G_full.nodes[node].update(G_core.nodes[node])
            for u, v in G_core.edges:
                if G_full.has_edge(u, v):
                    G_full[u][v].update(G_core[u][v])

            graph_list_core.append(G_core)
            graph_list_full.append(G_full)

        graph_dict[label] = {'core': graph_list_core, 'full': graph_list_full}

    high_force_edge_records, threshold_info = label_high_force_edges(
        graph_dict,
        use_global=USE_GLOBAL_HIGH_FORCE_THRESHOLD,
        use_core_for_threshold=USE_CORE_FOR_THRESHOLD,
        quantile_threshold=None,
        reference_geometry_label=REFERENCE_GEOMETRY_LABEL,
    )

    with open(os.path.join(PIPELINE_OUT_PATH, 'graph_dict_labeled.pkl'), 'wb') as handle:
        pickle.dump(graph_dict, handle)
    pd.DataFrame(high_force_edge_records).to_csv(os.path.join(PIPELINE_OUT_PATH, 'high_force_edges.csv'), index=False)
    _write_simulation_slices_txt(slice_df, PIPELINE_OUT_PATH)

    with open(os.path.join(PIPELINE_OUT_PATH, 'high_force_threshold_info.pkl'), 'wb') as handle:
        pickle.dump(threshold_info, handle)

    pd.DataFrame(pair_edge_index_records).to_csv(os.path.join(PIPELINE_OUT_PATH, 'pair_edge_connectivity_index.csv'), index=False)

    df_nodes, df_edges, df_graphs, df_graph_arrays = _build_feature_tables(graph_dict)
    df_nodes.to_csv(os.path.join(PIPELINE_OUT_PATH, 'node_features.csv'), index=False)
    df_edges.to_csv(os.path.join(PIPELINE_OUT_PATH, 'edge_features.csv'), index=False)
    df_graphs.to_csv(os.path.join(PIPELINE_OUT_PATH, 'graph_features.csv'), index=False)

    df_graph_arrays_serialized = df_graph_arrays.copy()
    for col in df_graph_arrays_serialized.columns:
        if col not in {'geometry', 'sim_idx'}:
            df_graph_arrays_serialized[col] = df_graph_arrays_serialized[col].map(_serialize_value)
    df_graph_arrays_serialized.to_csv(os.path.join(PIPELINE_OUT_PATH, 'graph_feature_arrays.csv'), index=False)
    df_graph_arrays.to_pickle(os.path.join(PIPELINE_OUT_PATH, 'graph_feature_arrays.pkl'))

    if PLOT_FORCE_DISTRIBUTIONS:
        plot_force_distributions_row(
            graph_dict,
            threshold_info,
            use_core_for_threshold=USE_CORE_FOR_THRESHOLD,
            out_dir=PIPELINE_OUT_PATH,
            bins=FORCE_DIST_BINS,
            logx=FORCE_DIST_LOGX,
            angle_order=angle_order,
            column_layout=FORCE_DIST_COLUMN,
            hist_linewidth=HIST_LINE_WIDTH,
            thr_linewidth=THRESHOLD_LINE_WIDTH,
            thr_color=THRESHOLD_LINE_COLOR,
        )

    print(f"Finalized staged graph outputs in: {PIPELINE_OUT_PATH}")


if __name__ == '__main__':
    main()
