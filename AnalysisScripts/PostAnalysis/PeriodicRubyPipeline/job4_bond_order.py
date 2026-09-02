#!/usr/bin/env python3
"""Job 4: Steinhardt bond order for ruby simulations and ideal crystals."""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import sph_harm
from pipeline_common import *
from crystal_common import STRUCTURES,crystal_graph,minimum_image_cell

def wigner_3j_equal_l(l,m1,m2,m3):
    """Racah factorial formula for integer (l l l; m1 m2 m3)."""
    if m1+m2+m3 or any(abs(m)>l for m in (m1,m2,m3)): return 0.0
    f=math.factorial
    delta=math.sqrt(f(l)**3/f(3*l+1))
    pref=(-1)**m3*delta*math.sqrt(math.prod(f(l+m)*f(l-m) for m in (m1,m2,m3)))
    zmin=max(0,-m1,m2); zmax=min(l,l-m1,l+m2); total=0.0
    for z in range(zmin,zmax+1):
        den=f(z)*f(l-z)*f(l-m1-z)*f(l+m2-z)*f(m1+z)*f(-m2+z)
        total+=(-1)**z/den
    return pref*total

def task_list(cfg): return [("ruby",a,s) for a in cfg["angles"] for s in range(cfg["simulations_per_angle"])]+[("crystal",s,0) for s in STRUCTURES]
def geometry(cfg):
    path=Path(cfg["output_root"])/"job0_metadata"/"geometry_estimate.json"
    if not path.exists(): raise FileNotFoundError("Run Job 0 before Job 4")
    return json.loads(path.read_text())

def bond_order(G,cfg,cell,periodic):
    nodes,pos=particle_positions(G); idx={n:i for i,n in enumerate(nodes)}; src=[];dst=[];vec=[]
    for u,v in G.edges():
        if u not in idx or v not in idx: continue
        raw=pos[idx[v]]-pos[idx[u]]; dv=minimum_image_cell(raw[None,:],cell,periodic)[0]
        src.extend([idx[u],idx[v]]); dst.extend([idx[v],idx[u]]); vec.extend([dv,-dv])
    src=np.asarray(src,int);dst=np.asarray(dst,int);vec=np.asarray(vec,float); norm=np.linalg.norm(vec,axis=1); unit=vec/norm[:,None]
    theta=np.arccos(np.clip(unit[:,2],-1,1)); phi=np.arctan2(unit[:,1],unit[:,0]); counts=np.bincount(src,minlength=len(nodes)); qlm={}; result={"node_id":nodes,"x":pos[:,0],"y":pos[:,1],"z":pos[:,2],"coordination":counts,"low_coordination":counts<cfg["low_coordination_threshold"]}
    globals_={}
    for l in cfg["bond_order_l"]:
        arr=np.zeros((len(nodes),2*l+1),complex)
        for mi,m in enumerate(range(-l,l+1)):
            y=sph_harm(m,l,phi,theta); np.add.at(arr[:,mi],src,y)
        valid=counts>0; arr[valid]/=counts[valid,None]; arr[~valid]=np.nan; qlm[l]=arr
        q=np.sqrt(4*np.pi/(2*l+1)*np.nansum(np.abs(arr)**2,axis=1)); q[~valid]=np.nan; result[f"q{l}"]=q
        coherent=np.mean(np.asarray([sph_harm(m,l,phi,theta) for m in range(-l,l+1)]),axis=1)
        globals_[f"Q{l}"]=float(np.sqrt(4*np.pi/(2*l+1)*np.sum(np.abs(coherent)**2))); globals_[f"mean_local_q{l}"]=float(np.nanmean(q))
    for l in cfg["neighbor_averaged_l"]:
        total=np.nan_to_num(qlm[l]).copy()
        for s,d in zip(src,dst): total[s]+=np.nan_to_num(qlm[l][d])
        averaged=total/(counts+1)[:,None]; qbar=np.sqrt(4*np.pi/(2*l+1)*np.sum(np.abs(averaged)**2,axis=1)); qbar[counts==0]=np.nan; result[f"qbar{l}"]=qbar
    for l in cfg["third_order_l"]:
        arr=qlm[l]; w=np.zeros(len(nodes),complex)
        for m1 in range(-l,l+1):
            for m2 in range(-l,l+1):
                m3=-m1-m2
                if abs(m3)<=l:
                    coef=wigner_3j_equal_l(l,m1,m2,m3)
                    if coef: w+=coef*arr[:,m1+l]*arr[:,m2+l]*arr[:,m3+l]
        denom=np.sum(np.abs(arr)**2,axis=1)**1.5; result[f"w{l}"]=w.real; result[f"what{l}"]=np.divide(w.real,denom,out=np.full(len(nodes),np.nan),where=denom>0)
    q6=qlm[6]; numerator=np.sum(q6[src]*np.conj(q6[dst]),axis=1).real; den=np.sqrt(np.sum(abs(q6[src])**2,axis=1)*np.sum(abs(q6[dst])**2,axis=1)); s6=np.divide(numerator,den,out=np.full(len(src),np.nan),where=den>0)
    incident_sum=np.zeros(len(nodes)); incident_n=np.zeros(len(nodes),int); good=np.isfinite(s6); np.add.at(incident_sum,src[good],s6[good]); np.add.at(incident_n,src[good],1)
    result["mean_s6"]=np.divide(incident_sum,incident_n,out=np.full(len(nodes),np.nan),where=incident_n>0); result["fraction_s6_above_threshold"]=np.array([np.mean(s6[(src==i)&good]>cfg["s6_threshold"]) if np.any((src==i)&good) else np.nan for i in range(len(nodes))])
    edge=pd.DataFrame({"source_node":[nodes[i] for i in src[::2]],"target_node":[nodes[i] for i in dst[::2]],"s6":s6[::2]})
    return pd.DataFrame(result),edge,globals_

def validate_crystals(cfg,root):
    rng=np.random.default_rng(cfg["random_seed"]); rows=[]
    for name in STRUCTURES:
        G=crystal_graph(name,cfg["particle_diameter"],cfg["contact_distance_tolerance_fraction"]); cell=np.asarray(G.graph["cell_matrix"]); periodic=tuple(G.graph["periodic_axes"])
        base,edges,system=bond_order(G,cfg,cell,periodic)
        # Proper random rotation applied to positions and cell together.
        q,_=np.linalg.qr(rng.normal(size=(3,3))); q[:,0]*=np.linalg.det(q)
        rotated=G.copy()
        for n,d in rotated.nodes(data=True): d["position"]=tuple(np.asarray(d["position"])@q)
        rot_cell=cell@q; rot,_,_=bond_order(rotated,cfg,rot_cell,periodic)
        # Periodic translation and wrapping in fractional coordinates.
        translated=G.copy(); shift=np.array([.371,.219,0.])
        for n,d in translated.nodes(data=True):
            frac=np.asarray(d["position"])@np.linalg.inv(cell)+shift
            for axis in periodic: frac[axis]%=1.0
            d["position"]=tuple(frac@cell)
        trans,_,_=bond_order(translated,cfg,cell,periodic)
        max_coord=int(base.coordination.max()); equivalent=base[base.coordination==max_coord]
        for prop in [f"q{l}" for l in cfg["bond_order_l"]]+["qbar4","qbar6","what4","what6"]:
            rows.append(dict(structure=name,property=prop,equivalent_site_std=float(equivalent[prop].std()),rotation_max_abs_error=float(np.nanmax(np.abs(base[prop]-rot[prop]))),translation_max_abs_error=float(np.nanmax(np.abs(base[prop]-trans[prop])))))
        rows.append(dict(structure=name,property="neighbor_validation",equivalent_site_std=np.nan,rotation_max_abs_error=abs(len(edges)-G.number_of_edges()),translation_max_abs_error=0.0))
        atomic_csv(root/"job4_bond_order"/f"validation_{name}_ideal_values.csv",base)
    frame=pd.DataFrame(rows); atomic_csv(root/"job4_bond_order"/"validation_errors.csv",frame)
    numeric=frame[["rotation_max_abs_error","translation_max_abs_error"]].to_numpy(float)
    passed=bool(np.nanmax(numeric)<=cfg["validation_tolerance"])
    atomic_json(root/"job4_bond_order"/"validation_status.json",{"passed":passed,"tolerance":cfg["validation_tolerance"],"maximum_error":float(np.nanmax(numeric))})
    if not passed: raise RuntimeError("Bond-order rotational/translation validation failed")
    return frame

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--task-id",type=int); ap.add_argument("--validate",action="store_true"); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config); root=ensure_layout(cfg)
    if args.validate:
        if args.dry_run: print("Would validate SC/BCC/FCC/HCP rotation and periodic translation invariance"); return
        validate_crystals(cfg,root); return
    tasks=task_list(cfg); tid=task_id(args.task_id)
    if tid is None: raise SystemExit("Provide --task-id or SLURM_ARRAY_TASK_ID")
    kind,label,sim=tasks[tid]; stem=f"ruby_{label}_sim{sim:03d}" if kind=="ruby" else f"crystal_{label}"; out=root/"job4_bond_order"/f"{stem}_nodes.csv"; log=setup_logging(cfg,"job4",tid)
    if output_valid(out,cfg,("node_id","q4","q6")): log.info("Skipping %s",out); return
    if args.dry_run: print(tid,kind,label,sim,out); return
    if kind=="ruby":
        graphs=load_graph_dict(cfg); G=graphs[label]["core"][sim]; geo=geometry(cfg); lengths=[geo["box_lengths"][str(i)] for i in range(3)]; cell=np.diag(lengths); periodic=tuple(geo["periodic_axes"])
    else: G=crystal_graph(label,cfg["particle_diameter"],cfg["contact_distance_tolerance_fraction"]); cell=np.asarray(G.graph["cell_matrix"]); periodic=tuple(G.graph["periodic_axes"])
    nodes,edges,global_=bond_order(G,cfg,cell,periodic); nodes.insert(0,"system",stem); edges.insert(0,"system",stem); atomic_csv(out,nodes); atomic_csv(out.with_name(out.name.replace("_nodes","_edges")),edges); atomic_json(out.with_name(out.name.replace("_nodes.csv","_system.json")),global_); mark_complete(out,cfg,{"kind":kind,"label":label,"sim":sim,"particles":len(nodes)}); log.info("Complete %s",out)
if __name__=="__main__":main()
