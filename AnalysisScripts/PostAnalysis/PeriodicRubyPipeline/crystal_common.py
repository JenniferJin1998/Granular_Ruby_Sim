"""Deterministic SC/BCC/FCC/HCP reference structures and contact graphs."""
import numpy as np
import networkx as nx

STRUCTURES=("SC","BCC","FCC","HCP")

def minimum_image_cell(vectors,cell,periodic=(0,1)):
    cell=np.asarray(cell,float); inv=np.linalg.inv(cell); frac=np.asarray(vectors)@inv
    for axis in periodic: frac[...,axis]-=np.round(frac[...,axis])
    return frac@cell

def crystal_positions(name,diameter):
    a=float(diameter)
    if name=="SC":
        dims=(16,17,6); basis=np.array([[0,0,0.]])
        lattice=np.diag([a,a,a])
    elif name=="BCC":
        # Nearest-neighbor distance sqrt(3)*a_cell/2 = diameter.
        ac=2*a/np.sqrt(3); dims=(11,12,6); lattice=np.diag([ac,ac,ac]); basis=np.array([[0,0,0.],[.5,.5,.5]])
    elif name=="FCC":
        ac=np.sqrt(2)*a; dims=(8,8,6); lattice=np.diag([ac,ac,ac]); basis=np.array([[0,0,0.],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    elif name=="HCP":
        # Rectangular representation of the ideal hexagonal lattice; six A/B
        # particle layers, even lateral repeat counts, ideal layer spacing.
        nx_,ny_,nz=18,14,6; dz=np.sqrt(2/3)*a; positions=[]
        for k in range(nz):
            layer_shift=np.array([.5*a,np.sqrt(3)*a/6,0]) if k%2 else np.zeros(3)
            for j in range(ny_):
                for i in range(nx_): positions.append(np.array([(i+.5*(j%2))*a,j*np.sqrt(3)*a/2,k*dz])+layer_shift)
        cell=np.diag([nx_*a,ny_*np.sqrt(3)*a/2,nz*dz]); return np.asarray(positions),cell,(0,1),dict(replication=[nx_,ny_,nz],layers=nz)
    else: raise ValueError(name)
    positions=[]
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                origin=np.array([i,j,k],float)
                for b in basis: positions.append((origin+b)@lattice)
    cell=np.diag(np.asarray(dims)@lattice); return np.asarray(positions),cell,(0,1),dict(replication=list(dims),layers=dims[2])

def crystal_graph(name,diameter,tolerance=.08):
    pos,cell,periodic,meta=crystal_positions(name,diameter); G=nx.Graph(structure=name,cell_matrix=cell.tolist(),periodic_axes=list(periodic),particle_diameter=diameter,**meta)
    for i,p in enumerate(pos): G.add_node(i,position=tuple(p),is_wall=False)
    # Chunked pair search avoids constructing one enormous N x N x 3 array.
    for i in range(len(pos)-1):
        delta=minimum_image_cell(pos[i+1:]-pos[i],cell,periodic); distance=np.linalg.norm(delta,axis=1)
        for offset in np.where(distance<=diameter*(1+tolerance))[0]:
            if distance[offset]>=diameter*(1-tolerance): G.add_edge(i,i+1+int(offset),distance=float(distance[offset]))
    volume=abs(np.linalg.det(cell)); sphere_volume=np.pi*diameter**3/6
    G.graph.update(particle_count=len(pos),box_volume=volume,packing_fraction=len(pos)*sphere_volume/volume,
                   neighbor_definition=f"minimum-image distance within ±{tolerance:.3f} of particle diameter")
    return G
