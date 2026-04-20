"""
Graph Generation and Analysis for Test Data (Single Simulation)
Tests with: Forces_periodic.mat, Pos_periodic.mat, Stresses_periodic.mat
"""

import os
import time
import numpy as np
import scipy.io
import networkx as nx
import pickle
from collections import defaultdict
from GraphRicciCurvature.OllivierRicci import OllivierRicci

# === Config ===
data_path = 'Data/Testing'
output_path = 'Data/Testing/results'
os.makedirs(output_path, exist_ok=True)

# Timing dictionary
timing = {}

# === Load Data ===
print("="*60)
print("LOADING TEST DATA")
print("="*60)

t_start = time.time()

# Load files
forces_file = os.path.join(data_path, 'Forces_periodic.mat')
pos_file = os.path.join(data_path, 'Pos_periodic.mat')
stresses_file = os.path.join(data_path, 'Stresses_periodic.mat')

print(f"Loading {forces_file}...")
forces_data = scipy.io.loadmat(forces_file)
print(f"  Keys: {list(forces_data.keys())}")

print(f"Loading {pos_file}...")
pos_data = scipy.io.loadmat(pos_file)
print(f"  Keys: {list(pos_data.keys())}")

print(f"Loading {stresses_file}...")
stresses_data = scipy.io.loadmat(stresses_file)
print(f"  Keys: {list(stresses_data.keys())}")

timing['data_loading'] = time.time() - t_start
print(f"\n✓ Data loading completed in {timing['data_loading']:.2f}s")

# === Extract Data ===
print("\n" + "="*60)
print("EXTRACTING DATA")
print("="*60)

# Extract force data
# Format: [num_contacts, columns] where columns are:
# timestep#, particle_ID1, particle_ID2, contact_x, contact_y, contact_z, 
# delta, delta_t, normal_force_mag, tangential_force_mag, 
# normal_force_unit_vector (3 components), tangential_force_unit_vector (3 components)
forces_key = [k for k in forces_data.keys() if not k.startswith('__')][0]
forces = forces_data[forces_key]
print(f"Forces data shape: {forces.shape}")
print(f"  Columns: timestep, pid1, pid2, cx, cy, cz, delta, delta_t, n_force, t_force, n_unit(3), t_unit(3)")
print(f"  Note: Negative particle ID indicates wall contact")

# Extract position data: [num_particles, 3] for x, y, z
pos_key = [k for k in pos_data.keys() if not k.startswith('__')][0]
positions = pos_data[pos_key]
print(f"Positions shape: {positions.shape}")
print(f"  Columns: x, y, z")

# Extract stresses data: [num_particles, 6] for e11, e22, e33, e23, e13, e12
stress_key = [k for k in stresses_data.keys() if not k.startswith('__')][0]
stresses = stresses_data[stress_key]
print(f"Stresses shape: {stresses.shape}")
print(f"  Columns: e11, e22, e33, e23, e13, e12")

# === Build Graphs (Full with walls and Core without walls) ===
print("\n" + "="*60)
print("BUILDING GRAPHS")
print("="*60)

t_start = time.time()

# Create FULL graph (includes wall contacts)
G_full = nx.Graph()

# Add particle nodes with positions and stress tensors
num_particles = positions.shape[0]
print(f"Number of particles: {num_particles}")

for i in range(num_particles):
    x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
    
    # Extract stress tensor components: e11, e22, e33, e23, e13, e12
    s11, s22, s33, s23, s13, s12 = stresses[i, 0], stresses[i, 1], stresses[i, 2], \
                                    stresses[i, 3], stresses[i, 4], stresses[i, 5]
    
    # Compute derived stress quantities
    hydro = (s11 + s22 + s33) / 3.0  # Hydrostatic stress
    vm = np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 
                 3*(s12**2 + s13**2 + s23**2))  # Von Mises stress
    
    G_full.add_node(i,
                   position=(float(x), float(y), float(z)),
                   x=float(x), y=float(y), z=float(z),
                   is_wall=False,
                   stress_11=float(s11), stress_22=float(s22), stress_33=float(s33),
                   stress_23=float(s23), stress_13=float(s13), stress_12=float(s12),
                   stress_vm=float(vm), stress_hydro=float(hydro))

print(f"✓ Added {G_full.number_of_nodes()} particle nodes")

# Add edges and wall nodes from force data
# Format: timestep, pid1, pid2, cx, cy, cz, delta, delta_t, n_force, t_force, n_unit(3), t_unit(3)
wall_node_counter = 0
wall_pid_map = {}

def get_node_id(pid, contact_idx, which_pid):
    """Handle wall contacts: negative particle ID creates wall node."""
    global wall_node_counter
    if pid < 0:
        key = (contact_idx, which_pid, pid)
        if key not in wall_pid_map:
            wall_id = f"wall_{wall_node_counter}"
            wall_pid_map[key] = wall_id
            wall_node_counter += 1
        else:
            wall_id = wall_pid_map[key]
        
        if wall_id not in G_full:
            G_full.add_node(wall_id, is_wall=True, wall_label=int(pid))
        return wall_id
    return int(pid)

print(f"\nProcessing {forces.shape[0]} contacts...")
num_wall_contacts = 0
num_particle_contacts = 0

for contact_idx in range(forces.shape[0]):
    row = forces[contact_idx]
    # timestep = row[0]  # Not used in graph
    pid1 = int(row[1])
    pid2 = int(row[2])
    
    node1 = get_node_id(pid1, contact_idx, 'pid1')
    node2 = get_node_id(pid2, contact_idx, 'pid2')
    
    if node1 in G_full and node2 in G_full:
        # Extract contact attributes
        contact_loc = (float(row[3]), float(row[4]), float(row[5]))
        delta = float(row[6])
        delta_t = float(row[7])
        normal_force = float(row[8])
        tangential_force = float(row[9])
        n_unit = (float(row[10]), float(row[11]), float(row[12]))
        t_unit = (float(row[13]), float(row[14]), float(row[15]))
        
        # Calculate angle with z-axis
        angle_deg = np.degrees(np.arccos(np.abs(np.clip(n_unit[2], -1.0, 1.0))))
        
        # Check if this is a wall contact
        is_wall_contact = G_full.nodes[node1].get('is_wall', False) or \
                         G_full.nodes[node2].get('is_wall', False)
        
        G_full.add_edge(node1, node2,
                       contact_location=contact_loc,
                       delta=delta,
                       delta_t=delta_t,
                       normal_force=normal_force,
                       tangential_force=tangential_force,
                       n_unit=n_unit,
                       t_unit=t_unit,
                       angle_with_zz=angle_deg,
                       is_wall_contact=is_wall_contact)
        
        if is_wall_contact:
            num_wall_contacts += 1
        else:
            num_particle_contacts += 1

print(f"✓ Added {G_full.number_of_edges()} total edges")
print(f"  - Particle-particle contacts: {num_particle_contacts}")
print(f"  - Wall contacts: {num_wall_contacts}")
print(f"  - Wall nodes created: {wall_node_counter}")

# Create CORE graph (particles only, no walls)
G_core = G_full.copy()
wall_nodes = [n for n, d in G_core.nodes(data=True) if d.get('is_wall', False)]
G_core.remove_nodes_from(wall_nodes)

print(f"\n✓ Core graph (no walls):")
print(f"  - Nodes: {G_core.number_of_nodes()}")
print(f"  - Edges: {G_core.number_of_edges()}")

timing['graph_construction'] = time.time() - t_start
print(f"\n✓ Graph construction completed in {timing['graph_construction']:.2f}s")

# === Basic Graph Properties ===
print("\n" + "="*60)
print("BASIC GRAPH PROPERTIES")
print("="*60)

t_start = time.time()

print("\nFULL GRAPH (with walls):")
print(f"  Nodes: {G_full.number_of_nodes()}")
print(f"  Edges: {G_full.number_of_edges()}")
print(f"  Density: {nx.density(G_full):.4f}")
print(f"  Connected: {nx.is_connected(G_full)}")

print("\nCORE GRAPH (particles only):")
print(f"  Nodes: {G_core.number_of_nodes()}")
print(f"  Edges: {G_core.number_of_edges()}")
print(f"  Density: {nx.density(G_core):.4f}")
print(f"  Connected: {nx.is_connected(G_core)}")

if nx.is_connected(G_core):
    print(f"  Average shortest path: {nx.average_shortest_path_length(G_core):.2f}")
    print(f"  Diameter: {nx.diameter(G_core)}")

timing['basic_properties'] = time.time() - t_start

# === Compute Centrality Metrics (on Core graph) ===
print("\n" + "="*60)
print("COMPUTING CENTRALITY METRICS (Core Graph)")
print("="*60)

t_start = time.time()

# Degree centrality
degree_cent = dict(G_core.degree())
for node in G_core.nodes():
    G_core.nodes[node]['degree'] = degree_cent[node]
print(f"✓ Degree centrality computed")

# Closeness centrality
closeness_cent = nx.closeness_centrality(G_core)
for node in G_core.nodes():
    G_core.nodes[node]['closeness'] = closeness_cent[node]
print(f"✓ Closeness centrality computed")

# Betweenness centrality
betweenness_cent = nx.betweenness_centrality(G_core)
for node in G_core.nodes():
    G_core.nodes[node]['betweenness'] = betweenness_cent[node]
print(f"✓ Betweenness centrality computed")

# Clustering coefficient
clustering_coef = nx.clustering(G_core)
for node in G_core.nodes():
    G_core.nodes[node]['clustering'] = clustering_coef[node]
print(f"✓ Clustering coefficient computed")

timing['centrality_metrics'] = time.time() - t_start
print(f"\n✓ Centrality metrics completed in {timing['centrality_metrics']:.2f}s")

# === Compute Ricci Curvature (on both graphs) ===
print("\n" + "="*60)
print("COMPUTING RICCI CURVATURE")
print("="*60)

t_start = time.time()

# Compute for FULL graph (with walls)
print("Computing Ollivier-Ricci curvature for FULL graph (alpha=0.5)...")
orc_full = OllivierRicci(G_full, alpha=0.5, verbose="ERROR")
orc_full.compute_ricci_curvature()

for u, v, d in orc_full.G.edges(data=True):
    if 'ricciCurvature' in d:
        G_full[u][v]['ricci_curvature'] = d['ricciCurvature']

print(f"✓ Edge curvature computed for {G_full.number_of_edges()} edges in full graph")

# Compute for CORE graph (particles only)
print("Computing Ollivier-Ricci curvature for CORE graph (alpha=0.5)...")
orc_core = OllivierRicci(G_core, alpha=0.5, verbose="ERROR")
orc_core.compute_ricci_curvature()

for u, v, d in orc_core.G.edges(data=True):
    if 'ricciCurvature' in d:
        G_core[u][v]['ricci_curvature'] = d['ricciCurvature']

print(f"✓ Edge curvature computed for {G_core.number_of_edges()} edges in core graph")

# Compute average node curvature for core graph
curv_sum = defaultdict(float)
count = defaultdict(int)
for u, v, data in G_core.edges(data=True):
    if 'ricci_curvature' in data:
        curv = data['ricci_curvature']
        curv_sum[u] += curv
        curv_sum[v] += curv
        count[u] += 1
        count[v] += 1

for node in G_core.nodes():
    if count[node] > 0:
        G_core.nodes[node]['avg_ricci_curvature'] = curv_sum[node] / count[node]
    else:
        G_core.nodes[node]['avg_ricci_curvature'] = 0.0

print(f"✓ Average node curvature computed for core graph")

timing['ricci_curvature'] = time.time() - t_start
print(f"\n✓ Ricci curvature completed in {timing['ricci_curvature']:.2f}s")

# === Statistics ===
print("\n" + "="*60)
print("STATISTICS SUMMARY")
print("="*60)

# Normal force statistics (from full graph)
normal_forces = [d['normal_force'] for u, v, d in G_full.edges(data=True) if 'normal_force' in d]
if normal_forces:
    print(f"\nNormal Force Statistics (Full Graph):")
    print(f"  Mean: {np.mean(normal_forces):.4e}")
    print(f"  Std: {np.std(normal_forces):.4e}")
    print(f"  Min: {np.min(normal_forces):.4e}")
    print(f"  Max: {np.max(normal_forces):.4e}")

# Tangential force statistics
tangential_forces = [d['tangential_force'] for u, v, d in G_full.edges(data=True) if 'tangential_force' in d]
if tangential_forces:
    print(f"\nTangential Force Statistics (Full Graph):")
    print(f"  Mean: {np.mean(tangential_forces):.4e}")
    print(f"  Std: {np.std(tangential_forces):.4e}")
    print(f"  Min: {np.min(tangential_forces):.4e}")
    print(f"  Max: {np.max(tangential_forces):.4e}")

# Curvature statistics (full graph)
curv_full = [d['ricci_curvature'] for u, v, d in G_full.edges(data=True) if 'ricci_curvature' in d]
if curv_full:
    print(f"\nRicci Curvature Statistics (Full Graph):")
    print(f"  Mean: {np.mean(curv_full):.4f}")
    print(f"  Std: {np.std(curv_full):.4f}")
    print(f"  Min: {np.min(curv_full):.4f}")
    print(f"  Max: {np.max(curv_full):.4f}")

# Curvature statistics (core graph)
curv_core = [d['ricci_curvature'] for u, v, d in G_core.edges(data=True) if 'ricci_curvature' in d]
if curv_core:
    print(f"\nRicci Curvature Statistics (Core Graph):")
    print(f"  Mean: {np.mean(curv_core):.4f}")
    print(f"  Std: {np.std(curv_core):.4f}")
    print(f"  Min: {np.min(curv_core):.4f}")
    print(f"  Max: {np.max(curv_core):.4f}")

# Von Mises stress statistics
vm_stress = [d['stress_vm'] for n, d in G_core.nodes(data=True) if 'stress_vm' in d]
if vm_stress:
    print(f"\nVon Mises Stress Statistics:")
    print(f"  Mean: {np.mean(vm_stress):.4e}")
    print(f"  Std: {np.std(vm_stress):.4e}")
    print(f"  Min: {np.min(vm_stress):.4e}")
    print(f"  Max: {np.max(vm_stress):.4e}")

# Hydrostatic stress statistics
hydro_stress = [d['stress_hydro'] for n, d in G_core.nodes(data=True) if 'stress_hydro' in d]
if hydro_stress:
    print(f"\nHydrostatic Stress Statistics:")
    print(f"  Mean: {np.mean(hydro_stress):.4e}")
    print(f"  Std: {np.std(hydro_stress):.4e}")
    print(f"  Min: {np.min(hydro_stress):.4e}")
    print(f"  Max: {np.max(hydro_stress):.4e}")

# === Save Results ===
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

t_start = time.time()

# Save full graph (with walls) as pickle
graph_full_pkl = os.path.join(output_path, 'test_graph_full.pkl')
with open(graph_full_pkl, 'wb') as f:
    pickle.dump(G_full, f)
print(f"✓ Full graph (with walls) saved to: {graph_full_pkl}")

# Save core graph (particles only) as pickle
graph_core_pkl = os.path.join(output_path, 'test_graph_core.pkl')
with open(graph_core_pkl, 'wb') as f:
    pickle.dump(G_core, f)
print(f"✓ Core graph (particles only) saved to: {graph_core_pkl}")

# Create GraphML-compatible copies (convert tuples to strings)
def make_graphml_compatible(G):
    """Convert tuple attributes to strings for GraphML export."""
    G_copy = G.copy()
    for node, data in G_copy.nodes(data=True):
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key + '_str'] = str(value)
                del data[key]
    for u, v, data in G_copy.edges(data=True):
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key + '_str'] = str(value)
                del data[key]
    return G_copy

print("\nConverting graphs for GraphML export...")
G_full_gml = make_graphml_compatible(G_full)
G_core_gml = make_graphml_compatible(G_core)

graph_full_graphml = os.path.join(output_path, 'test_graph_full.graphml')
nx.write_graphml(G_full_gml, graph_full_graphml)
print(f"✓ Full graph (with walls) saved to: {graph_full_graphml}")

graph_core_graphml = os.path.join(output_path, 'test_graph_core.graphml')
nx.write_graphml(G_core_gml, graph_core_graphml)
print(f"✓ Core graph (particles only) saved to: {graph_core_graphml}")

timing['saving'] = time.time() - t_start

# === Timing Summary ===
print("\n" + "="*60)
print("TIMING SUMMARY")
print("="*60)

total_time = sum(timing.values())

print(f"\nDetailed Timing:")
for step, duration in timing.items():
    percentage = (duration / total_time) * 100
    print(f"  {step:25s}: {duration:8.2f}s ({percentage:5.1f}%)")

print(f"\n{'='*60}")
print(f"TOTAL TIME: {total_time:.2f}s")
print(f"{'='*60}")

# Save timing info
timing_output = os.path.join(output_path, 'timing_info.txt')
with open(timing_output, 'w') as f:
    f.write("GRAPH ANALYSIS TIMING SUMMARY\n")
    f.write("="*60 + "\n\n")
    for step, duration in timing.items():
        percentage = (duration / total_time) * 100
        f.write(f"{step:25s}: {duration:8.2f}s ({percentage:5.1f}%)\n")
    f.write("\n" + "="*60 + "\n")
    f.write(f"TOTAL TIME: {total_time:.2f}s\n")

print(f"\n✓ Timing info saved to: {timing_output}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
