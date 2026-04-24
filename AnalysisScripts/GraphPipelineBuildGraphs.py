import os
import pandas as pd
import numpy as np

from GraphPipelineCommon import (
    RAW_GRAPH_DIR,
    base_path,
    _build_full_graph_from_slice,
    _discover_geometry_folders,
    _find_mat_file,
    _infer_particles_per_simulation,
    _load_mat_array,
    _prepare_roi_array,
    _raw_graph_path,
    _save_pickle,
    MAX_SIMS_PER_GEOMETRY,
)


def main():
    os.makedirs(RAW_GRAPH_DIR, exist_ok=True)
    slice_records = []

    geometry_specs = _discover_geometry_folders(base_path)
    print(f"Building raw graphs for: {', '.join(label for label, _ in geometry_specs)}")

    for label, folder in geometry_specs:
        folder_path = os.path.join(base_path, folder)
        print(f"Processing {label} from {folder_path}")

        force_file = _find_mat_file(folder_path, ('Forces.mat', 'forces.mat'), ('force',))
        count_file = _find_mat_file(folder_path, ('flengths.mat',), ('length',))
        pos_file = _find_mat_file(folder_path, ('Pos.mat', 'pos.mat'), ('pos',))
        stress_file = _find_mat_file(folder_path, ('AllStresses.mat',), ('stress',))
        roi_file = _find_mat_file(folder_path, ('ROI.mat', 'roi.mat'), ('roi',), optional=True)

        contact_forces = _load_mat_array(force_file, ('forces_collect',), ('force',), 'force data')
        contact_counts = np.asarray(_load_mat_array(count_file, ('f_lengths',), ('length',), 'force length data')).reshape(-1)
        particle_positions = np.asarray(_load_mat_array(pos_file, ('Pos_collect',), ('pos',), 'position data'))
        stress_components = np.asarray(_load_mat_array(stress_file, ('sigma_collect',), ('sigma', 'stress'), 'stress data'))

        roi_array = None
        if roi_file is not None:
            try:
                roi_array = _load_mat_array(roi_file, ('ROI',), ('roi',), 'ROI data')
            except Exception as exc:
                print(f"Error loading ROI for {label}: {exc}")

        num_particles_per_sim = _infer_particles_per_simulation(particle_positions, contact_counts, label)
        total_particles = num_particles_per_sim * len(contact_counts)
        roi_array = _prepare_roi_array(roi_array, total_particles, label)

        print(
            f"{label}: {len(contact_counts)} simulations, "
            f"{num_particles_per_sim} particles/simulation, {int(np.sum(contact_counts))} contacts total."
        )

        start_idx_contact = 0
        start_idx_particle = 0
        for sim_idx, num_contacts in enumerate(contact_counts):
            if MAX_SIMS_PER_GEOMETRY > 0 and sim_idx >= MAX_SIMS_PER_GEOMETRY:
                break
            num_contacts = int(num_contacts)
            G_full = _build_full_graph_from_slice(
                label=label,
                sim_idx=sim_idx,
                force_data=np.asarray(contact_forces[start_idx_contact:start_idx_contact + num_contacts]).copy(),
                pos_array=np.asarray(particle_positions[start_idx_particle:start_idx_particle + num_particles_per_sim]).copy(),
                stress_array=np.asarray(stress_components[start_idx_particle:start_idx_particle + num_particles_per_sim]).copy(),
                roi_labels=np.asarray(roi_array[start_idx_particle:start_idx_particle + num_particles_per_sim]).copy(),
                num_contacts=num_contacts,
                num_particles=int(num_particles_per_sim),
            )
            graph_path = _raw_graph_path(label, sim_idx)
            _save_pickle(G_full, graph_path)
            slice_records.append({
                'label': label,
                'sim_idx': sim_idx,
                'start_idx_contact': int(start_idx_contact),
                'num_contacts': num_contacts,
                'end_idx_contact': int(start_idx_contact + num_contacts),
                'start_idx_particle': int(start_idx_particle),
                'num_particles': int(num_particles_per_sim),
                'end_idx_particle': int(start_idx_particle + num_particles_per_sim),
                'raw_graph_file': graph_path,
            })
            print(f"[{label} sim {sim_idx:03d}] saved raw graph: {graph_path}")
            start_idx_contact += num_contacts
            start_idx_particle += num_particles_per_sim

    slices_path = os.path.join(os.path.dirname(RAW_GRAPH_DIR), 'simulation_slices.csv')
    pd.DataFrame(slice_records).to_csv(slices_path, index=False)
    print(f"Saved slice index: {slices_path}")


if __name__ == '__main__':
    main()
