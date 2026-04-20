"""
Quick data inspection script to understand the test data structure
"""

import numpy as np
import scipy.io

print("="*60)
print("DATA INSPECTION")
print("="*60)

# Load forces
print("\n1. FORCES DATA:")
forces_data = scipy.io.loadmat('Data/Testing/Forces_periodic.mat')
forces = forces_data['Forces']
print(f"   Shape: {forces.shape}")
print(f"   Format: [timestep, pid1, pid2, cx, cy, cz, delta, delta_t, n_force, t_force, n_unit(3), t_unit(3)]")

# Check timesteps
timesteps = forces[:, 0]
unique_timesteps = np.unique(timesteps)
print(f"\n   Timesteps:")
print(f"     Unique: {len(unique_timesteps)}")
print(f"     Values: {unique_timesteps}")

# Check for duplicates
pid1 = forces[:, 1].astype(int)
pid2 = forces[:, 2].astype(int)
pairs = list(zip(pid1, pid2))
unique_pairs = set()
symmetric_pairs = set()
duplicate_count = 0

for i, (p1, p2) in enumerate(pairs):
    # Normalize pair (undirected: (1,2) == (2,1))
    normalized = tuple(sorted([p1, p2]))
    if normalized in unique_pairs:
        duplicate_count += 1
    else:
        unique_pairs.add(normalized)
    
    # Check if we have both directions
    if (p2, p1) in pairs[:i]:
        symmetric_pairs.add(normalized)

print(f"\n   Contact pair analysis:")
print(f"     Total records: {len(pairs)}")
print(f"     Unique pairs (treating i-j same as j-i): {len(unique_pairs)}")
print(f"     Duplicate/repeated pairs: {duplicate_count}")
print(f"     Pairs with both directions (i→j and j→i): {len(symmetric_pairs)}")

# Check wall contacts
wall_contacts = np.sum((pid1 < 0) | (pid2 < 0))
particle_contacts = len(pairs) - wall_contacts
print(f"\n   Contact types:")
print(f"     Particle-particle records: {particle_contacts}")
print(f"     Wall contact records: {wall_contacts}")
print(f"     Total: {len(pairs)}")

# Sample some data
print(f"\n   Sample forces (first 5):")
for i in range(min(5, forces.shape[0])):
    print(f"     {i}: ts={forces[i,0]:.0f}, pid1={int(forces[i,1])}, pid2={int(forces[i,2])}, "
          f"n_force={forces[i,8]:.4e}, t_force={forces[i,9]:.4e}")

# Load positions
print("\n2. POSITIONS DATA:")
pos_data = scipy.io.loadmat('Data/Testing/Pos_periodic.mat')
pos = pos_data['Pos_collect']
print(f"   Shape: {pos.shape}")
print(f"   Format: [x, y, z]")
print(f"   Sample (first particle): x={pos[0,0]:.4f}, y={pos[0,1]:.4f}, z={pos[0,2]:.4f}")

# Load stresses
print("\n3. STRESSES DATA:")
stress_data = scipy.io.loadmat('Data/Testing/Stresses_periodic.mat')
stress = stress_data['sigma_collect']
print(f"   Shape: {stress.shape}")
print(f"   Format: [e11, e22, e33, e23, e13, e12]")
print(f"   Sample (first particle):")
print(f"     e11={stress[0,0]:.4e}, e22={stress[0,1]:.4e}, e33={stress[0,2]:.4e}")
print(f"     e23={stress[0,3]:.4e}, e13={stress[0,4]:.4e}, e12={stress[0,5]:.4e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Particles: {pos.shape[0]}")
print(f"Contact records: {forces.shape[0]}")
print(f"Unique particle pairs: {len(unique_pairs)}")
print(f"Expected graph edges: ~{len(unique_pairs)} (if data is deduplicated)")
print(f"Discrepancy: {forces.shape[0] - len(unique_pairs)} duplicate/repeated contacts")
print("="*60)
