#!/usr/bin/env python3
"""Job 6: periodic-aware connected high-force contact clusters."""
import argparse,json
from collections import Counter,deque
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from pipeline_common import *


def tasks(graphs,cfg): return graph_index(graphs,cfg)


def geometry(cfg,root):
    path=root/"job0_metadata"/"geometry_estimate.json"
    if not path.exists(): raise FileNotFoundError(f"Job 0 geometry prerequisite missing: {path}")
    geo=json.loads(path.read_text()); return {int(k):float(v) for k,v in geo["box_lengths"].items()},[int(x) for x in geo["periodic_axes"]]


def unwrap_component(H,box_lengths,periodic_axes):
    root=next(iter(H)); raw={n:np.asarray(H.nodes[n]["position"],float) for n in H}; unwrapped={root:raw[root].copy()}; queue=deque([root])
    while queue:
        u=queue.popleft()
        for v in H.neighbors(u):
            if v in unwrapped: continue
            delta=minimum_image(raw[v]-raw[u],box_lengths,periodic_axes); unwrapped[v]=unwrapped[u]+delta; queue.append(v)
    return unwrapped


def json_counts(values): return json.dumps({str(k):int(v) for k,v in sorted(Counter(values).items())},sort_keys=True)


def cluster_row(H,cluster_id,G,box_lengths,periodic_axes,cfg,domain_min,domain_max):
    n=H.number_of_nodes(); e=H.number_of_edges(); coords=unwrap_component(H,box_lengths,periodic_axes); xyz=np.asarray([coords[x] for x in H],float); center=xyz.mean(0); centered=xyz-center; gyration=centered.T@centered/n
    eigvals,eigvecs=np.linalg.eigh(gyration); order=np.argsort(eigvals)[::-1]; eigvals=np.maximum(eigvals[order],0); eigvecs=eigvecs[:,order]; principal=eigvecs[:,0]; rg=float(np.sqrt(eigvals.sum())); aspect=float(np.sqrt(eigvals[0]/eigvals[-1])) if eigvals[-1]>np.finfo(float).eps*max(eigvals[0],1.0) else np.nan
    degrees=[d for _,d in H.degree()]; cycles=nx.cycle_basis(H); cycle_sizes=[len(c) for c in cycles]; loading=int(cfg["loading_axis"]); raw=np.asarray([G.nodes[x]["position"] for x in H],float)
    nonperiodic=[axis for axis in range(3) if axis not in periodic_axes]; boundary=[]
    for axis in nonperiodic: boundary.extend((raw[:,axis]-domain_min[axis]).tolist()); boundary.extend((domain_max[axis]-raw[:,axis]).tolist())
    loading_dist=np.minimum(raw[:,loading]-domain_min[loading],domain_max[loading]-raw[:,loading]) if loading not in periodic_axes else np.full(n,np.nan)
    try: diameter=nx.diameter(H) if n>1 else 0; mean_path=nx.average_shortest_path_length(H) if n>1 else 0.0
    except nx.NetworkXError: diameter=mean_path=np.nan
    try: assort=nx.degree_assortativity_coefficient(H) if e>1 else np.nan
    except Exception: assort=np.nan
    edge_conn=np.nan
    if 1<n<=cfg.get("job2_exact_connectivity_max_nodes",250):
        try: edge_conn=nx.edge_connectivity(H)
        except Exception: pass
    spectral_radius=algebraic=np.nan
    if n>1:
        try:
            A=nx.to_scipy_sparse_array(H,dtype=float,format="csr")
            if n<=12:
                spectral_radius=float(np.max(np.abs(np.linalg.eigvalsh(A.toarray())))); algebraic=float(np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(H).toarray()))[1])
            else:
                spectral_radius=float(abs(eigsh(A,k=1,which="LM",return_eigenvectors=False)[0])); algebraic=float(np.sort(eigsh(nx.laplacian_matrix(H).astype(float),k=2,which="SM",return_eigenvectors=False))[1])
        except Exception: pass
    core_numbers=nx.core_number(H) if e else {x:0 for x in H}; max_core=max(core_numbers.values(),default=0); largest_core=sum(v==max_core for v in core_numbers.values()) if max_core else 0
    normals=finite([d.get("normal_force") for *_,d in H.edges(data=True)]); tangentials=finite([d.get("tangential_force") for *_,d in H.edges(data=True)]); stress=finite([d.get("stress_vm") for _,d in H.nodes(data=True)])
    orientations=[]
    for *_,d in H.edges(data=True):
        unit=np.asarray(d.get("n_unit",[np.nan]*3),float)
        if unit.shape==(3,) and np.isfinite(unit).all() and np.linalg.norm(unit)>0: orientations.append(abs(unit[loading]/np.linalg.norm(unit)))
    orientation=np.asarray(orientations,float); align=float(abs(principal[loading])); ext=np.ptp(xyz,axis=0)
    return dict(cluster_id=cluster_id,node_count=n,edge_count=e,extent_x=ext[0],extent_y=ext[1],extent_z=ext[2],radius_of_gyration=rg,aspect_ratio=aspect,
                principal_axis_x=principal[0],principal_axis_y=principal[1],principal_axis_z=principal[2],principal_axis_loading_alignment=align,
                distance_from_boundary=min(boundary) if boundary else np.nan,distance_from_loading_surface=float(np.min(loading_dist)) if np.isfinite(loading_dist).any() else np.nan,
                mean_degree=float(np.mean(degrees)),degree_distribution=json_counts(degrees),density=nx.density(H),diameter=diameter,average_shortest_path_length=mean_path,
                clustering_coefficient=float(np.mean(list(nx.clustering(H).values()))),triangles=int(sum(nx.triangles(H).values())//3),loop_count=len(cycles),loop_size_distribution=json_counts(cycle_sizes),
                assortativity=assort,edge_connectivity=edge_conn,spectral_radius=spectral_radius,algebraic_connectivity=algebraic,branches=sum(d>=3 for d in degrees),terminal_nodes=sum(d==1 for d in degrees),
                max_core_number=max_core,largest_core_fraction=largest_core/n if n else np.nan,diameter_per_sqrt_node=diameter/np.sqrt(n) if n and np.isfinite(diameter) else np.nan,loops_per_node=len(cycles)/n if n else np.nan,
                mean_normal_force=np.mean(normals) if len(normals) else np.nan,max_normal_force=np.max(normals) if len(normals) else np.nan,total_normal_force=np.sum(normals) if len(normals) else np.nan,
                mean_tangential_force=np.mean(tangentials) if len(tangentials) else np.nan,max_tangential_force=np.max(tangentials) if len(tangentials) else np.nan,
                mean_particle_stress=np.mean(stress) if len(stress) else np.nan,max_particle_stress=np.max(stress) if len(stress) else np.nan,
                mean_force_loading_alignment=np.mean(orientation) if len(orientation) else np.nan,mean_force_angle_to_loading_deg=np.degrees(np.arccos(np.clip(orientation,0,1))).mean() if len(orientation) else np.nan,
                force_angle_to_loading_std_deg=np.degrees(np.arccos(np.clip(orientation,0,1))).std(ddof=1) if len(orientation)>1 else np.nan,
                node_ids=json.dumps([int(x) if isinstance(x,(int,np.integer)) else str(x) for x in H.nodes()]))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--task-id",type=int); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config); root=ensure_layout(cfg); graphs=load_graph_dict(cfg); all_tasks=tasks(graphs,cfg); tid=task_id(args.task_id)
    if tid is None: raise SystemExit("Provide --task-id or SLURM_ARRAY_TASK_ID")
    if tid<0 or tid>=len(all_tasks): raise SystemExit(f"task id must be 0..{len(all_tasks)-1}")
    angle,sim=all_tasks[tid]; out=root/"job6_force_clusters"/f"{angle}_sim{sim:03d}_clusters.csv"; summary_out=out.with_name(out.stem.replace("_clusters","")+"_simulation_summary.csv"); log=setup_logging(cfg,"job6",tid)
    if output_valid(out,cfg,("angle","sim_idx","cluster_id","node_count")): log.info("Skipping valid %s",out); return
    if args.dry_run: print(tid,angle,sim,out); return
    G=graphs[angle][cfg["primary_graph_view"]][sim]; _,edge_labels,_=high_force_labels(G,cfg); high_edges=[(u,v) for (u,v),flag in edge_labels.items() if flag]; F=nx.Graph(); F.add_edges_from(high_edges)
    for n in F: F.nodes[n].update(G.nodes[n])
    for u,v in F.edges(): F.edges[u,v].update(G.edges[u,v])
    box_lengths,periodic_axes=geometry(cfg,root); all_pos=np.asarray([d["position"] for _,d in G.nodes(data=True)],float); dmin=all_pos.min(0); dmax=all_pos.max(0); rows=[]
    components=sorted(nx.connected_components(F),key=lambda c:(-len(c),min(map(str,c))))
    for cid,nodes in enumerate(components): rows.append(cluster_row(F.subgraph(nodes).copy(),cid,G,box_lengths,periodic_axes,cfg,dmin,dmax))
    clusters=pd.DataFrame(rows); clusters.insert(0,"load_step",cfg["load_step"]); clusters.insert(0,"sim_idx",sim); clusters.insert(0,"angle",angle)
    sizes=clusters.node_count.to_numpy() if len(clusters) else np.array([]); diameters=clusters.diameter.to_numpy() if len(clusters) else np.array([]); clustered_nodes=F.number_of_nodes()
    summary=pd.DataFrame([dict(angle=angle,sim_idx=sim,load_step=cfg["load_step"],cluster_count=len(clusters),particle_fraction_in_clusters=clustered_nodes/G.number_of_nodes(),contact_fraction_in_clusters=len(high_edges)/G.number_of_edges() if G.number_of_edges() else np.nan,
                              largest_cluster_size=int(sizes.max()) if len(sizes) else 0,largest_cluster_fraction=float(sizes.max()/G.number_of_nodes()) if len(sizes) else 0.0,mean_cluster_size=float(sizes.mean()) if len(sizes) else np.nan,median_cluster_size=float(np.median(sizes)) if len(sizes) else np.nan,
                              cluster_size_distribution=json_counts(sizes.astype(int)),cluster_diameter_distribution=json_counts(finite(diameters).astype(int)),isolated_particles_included=False)])
    atomic_csv(out,clusters); atomic_csv(summary_out,summary); mark_complete(out,cfg,{"angle":angle,"sim_idx":sim,"clusters":len(clusters),"periodic_axes":periodic_axes}); log.info("Complete %s with %d clusters",out,len(clusters))


if __name__=="__main__": main()
