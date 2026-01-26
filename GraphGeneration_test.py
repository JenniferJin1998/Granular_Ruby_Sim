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

# Extract force data (assuming similar structure to forces.mat)
# forces_periodic.mat should contain edge list and force values
forces_key = [k for k in forces_data.keys() if not k.startswith('__')][0]
forces = forces_data[forces_key]
print(f"Forces data shape: {forces.shape}")
print(f"Forces data type: {forces.dtype}")

# Extract position data  
pos_key = [k for k in pos_data.keys() if not k.startswith('__')][0]
positions = pos_data[pos_key]
print(f"Positions shape: {positions.shape}")

# Extract stresses data
stress_key = [k for k in stresses_data.keys() if not k.startswith('__')][0]
stresses = stresses_data[stress_key]
print(f"Stresses shape: {stresses.shape}")

# === Build Graph ===
print("\n" + "="*60)
print("BUILDING GRAPH")
print("="*60)

t_start = time.time()

G = nx.Graph()

# Add nodes with positions
num_particles = positions.shape[0]
print(f"Number of particles: {num_particles}")

for i in range(num_particles):
    # Assuming positions is [N, 3] for x, y, z
    if positions.shape[1] >= 3:
        x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
    else:
        x, y = positions[i, 0], positions[i, 1]
        z = 0.0
    
    # Add node attributes
    G.add_node(i, 
               x=float(x), 
               y=float(y), 
               z=float(z),
               pos=(float(x), float(y), float(z)))
    
    # Add stress if available
    if i < stresses.shape[0]:
        # Assuming stress is scalar or first component
        if stresses.ndim == 1:
            stress_val = stresses[i]
        else:
            stress_val = stresses[i, 0] if stresses.shape[1] > 0 else 0.0
        G.nodes[i]['stress'] = float(stress_val)

print(f"✓ Added {G.number_of_nodes()} nodes")

# Add edges with forces
# Assuming forces contains [node1, node2, force_magnitude] or similar structure
print(f"\nProcessing edges from forces data...")

if forces.ndim == 2:
    if forces.shape[1] >= 3:
        # Format: [node1, node2, force_x, force_y, force_z] or [node1, node2, force]
        for i in range(forces.shape[0]):
            node1 = int(forces[i, 0])
            node2 = int(forces[i, 1])
            
            # Calculate force magnitude
            if forces.shape[1] >= 5:  # Has x, y, z components
                fx, fy, fz = forces[i, 2], forces[i, 3], forces[i, 4]
                force_mag = np.sqrt(fx**2 + fy**2 + fz**2)
            else:  # Just magnitude
                force_mag = forces[i, 2]
            
            # Add edge
            if node1 < num_particles and node2 < num_particles:
                G.add_edge(node1, node2, force=float(force_mag))

print(f"✓ Added {G.number_of_edges()} edges")

timing['graph_construction'] = time.time() - t_start
print(f"\n✓ Graph construction completed in {timing['graph_construction']:.2f}s")

# === Basic Graph Properties ===
print("\n" + "="*60)
print("BASIC GRAPH PROPERTIES")
print("="*60)

t_start = time.time()

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Connected: {nx.is_connected(G)}")

if nx.is_connected(G):
    print(f"Average shortest path: {nx.average_shortest_path_length(G):.2f}")
    print(f"Diameter: {nx.diameter(G)}")

timing['basic_properties'] = time.time() - t_start

# === Compute Centrality Metrics ===
print("\n" + "="*60)
print("COMPUTING CENTRALITY METRICS")
print("="*60)

t_start = time.time()

# Degree centrality
degree_cent = dict(G.degree())
for node in G.nodes():
    G.nodes[node]['degree'] = degree_cent[node]
print(f"✓ Degree centrality computed")

# Closeness centrality
closeness_cent = nx.closeness_centrality(G)
for node in G.nodes():
    G.nodes[node]['closeness'] = closeness_cent[node]
print(f"✓ Closeness centrality computed")

# Betweenness centrality
betweenness_cent = nx.betweenness_centrality(G)
for node in G.nodes():
    G.nodes[node]['betweenness'] = betweenness_cent[node]
print(f"✓ Betweenness centrality computed")

# Clustering coefficient
clustering_coef = nx.clustering(G)
for node in G.nodes():
    G.nodes[node]['clustering'] = clustering_coef[node]
print(f"✓ Clustering coefficient computed")

timing['centrality_metrics'] = time.time() - t_start
print(f"\n✓ Centrality metrics completed in {timing['centrality_metrics']:.2f}s")

# === Compute Ricci Curvature ===
print("\n" + "="*60)
print("COMPUTING RICCI CURVATURE")
print("="*60)

t_start = time.time()

print("Computing Ollivier-Ricci curvature (alpha=0.5)...")
orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
orc.compute_ricci_curvature()

# Extract edge curvatures
for u, v, d in orc.G.edges(data=True):
    if 'ricciCurvature' in d:
        G[u][v]['ricci_curvature'] = d['ricciCurvature']

print(f"✓ Edge curvature computed for {G.number_of_edges()} edges")

# Compute average node curvature
curv_sum = defaultdict(float)
count = defaultdict(int)
for u, v, data in G.edges(data=True):
    if 'ricci_curvature' in data:
        curv = data['ricci_curvature']
        curv_sum[u] += curv
        curv_sum[v] += curv
        count[u] += 1
        count[v] += 1

for node in G.nodes():
    if count[node] > 0:
        G.nodes[node]['avg_ricci_curvature'] = curv_sum[node] / count[node]
    else:
        G.nodes[node]['avg_ricci_curvature'] = 0.0

print(f"✓ Average node curvature computed")

timing['ricci_curvature'] = time.time() - t_start
print(f"\n✓ Ricci curvature completed in {timing['ricci_curvature']:.2f}s")

# === Statistics ===
print("\n" + "="*60)
print("STATISTICS SUMMARY")
print("="*60)

# Force statistics
forces_list = [d['force'] for u, v, d in G.edges(data=True) if 'force' in d]
if forces_list:
    print(f"\nForce Statistics:")
    print(f"  Mean: {np.mean(forces_list):.4f}")
    print(f"  Std: {np.std(forces_list):.4f}")
    print(f"  Min: {np.min(forces_list):.4f}")
    print(f"  Max: {np.max(forces_list):.4f}")

# Curvature statistics
curv_list = [d['ricci_curvature'] for u, v, d in G.edges(data=True) if 'ricci_curvature' in d]
if curv_list:
    print(f"\nRicci Curvature Statistics:")
    print(f"  Mean: {np.mean(curv_list):.4f}")
    print(f"  Std: {np.std(curv_list):.4f}")
    print(f"  Min: {np.min(curv_list):.4f}")
    print(f"  Max: {np.max(curv_list):.4f}")

# Stress statistics
stress_list = [d['stress'] for n, d in G.nodes(data=True) if 'stress' in d]
if stress_list:
    print(f"\nStress Statistics:")
    print(f"  Mean: {np.mean(stress_list):.4f}")
    print(f"  Std: {np.std(stress_list):.4f}")
    print(f"  Min: {np.min(stress_list):.4f}")
    print(f"  Max: {np.max(stress_list):.4f}")

# === Save Results ===
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

t_start = time.time()

# Save graph as pickle
graph_output = os.path.join(output_path, 'test_graph.pkl')
with open(graph_output, 'wb') as f:
    pickle.dump(G, f)
print(f"✓ Graph saved to: {graph_output}")

# Save graph as GraphML
graphml_output = os.path.join(output_path, 'test_graph.graphml')
nx.write_graphml(G, graphml_output)
print(f"✓ Graph saved to: {graphml_output}")

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
