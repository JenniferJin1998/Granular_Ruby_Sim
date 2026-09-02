#!/usr/bin/env python3
"""Job 2: one restartable simulation x hop task, parallelized over centers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from pipeline_common import *

_WORK_GRAPH=None; _WORK_CFG=None; _WORK_HOP=None; _WORK_BUNDLE=None
def _init_worker(G,cfg,hop,bundle):
    global _WORK_GRAPH,_WORK_CFG,_WORK_HOP,_WORK_BUNDLE; _WORK_GRAPH=G; _WORK_CFG=cfg; _WORK_HOP=hop; _WORK_BUNDLE=bundle
def _center(center):
    nodes=nx.single_source_shortest_path_length(_WORK_GRAPH,center,cutoff=_WORK_HOP).keys(); H=_WORK_GRAPH.subgraph(nodes)
    row={"center_node":center,**safe_graph_metrics(H,_WORK_CFG,_WORK_BUNDLE)}
    if _WORK_BUNDLE=="fast":
        for attr in ("stress_vm","stress_hydro","high_force_degree","nfd","nfd_r2"):
            vals=np.asarray([d.get(attr,np.nan) for _,d in H.nodes(data=True)],float); row[f"mean_{attr}"]=float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        forces=np.asarray([d.get("normal_force",np.nan) for *_,d in H.edges(data=True)],float); row["mean_normal_force"]=float(np.nanmean(forces)) if np.isfinite(forces).any() else np.nan
    return row

def tasks(graphs,cfg): return [(a,s,h,b) for b in cfg["job2_property_bundles"] for a,s in graph_index(graphs,cfg) for h in cfg["hop_sizes"]]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--task-id",type=int); ap.add_argument("--workers",type=int); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config); root=ensure_layout(cfg); graphs=load_graph_dict(cfg); all_tasks=tasks(graphs,cfg); tid=task_id(args.task_id)
    if tid is None: raise SystemExit("Provide --task-id or SLURM_ARRAY_TASK_ID")
    if tid<0 or tid>=len(all_tasks): raise SystemExit(f"task id must be 0..{len(all_tasks)-1}")
    angle,sim,hop,bundle=all_tasks[tid]; out=root/"job2_subgraphs"/f"{angle}_sim{sim:03d}_hop{hop}_{bundle}.csv"; log=setup_logging(cfg,"job2",tid)
    if output_valid(out,cfg,("center_node","mean_degree")): log.info("Skipping valid %s",out); return
    if args.dry_run: print(tid,angle,sim,hop,bundle,out); return
    G=graphs[angle]["core"][sim]; centers=list(G.nodes()); workers=args.workers or cfg["local_workers"]
    if workers>1:
        with ProcessPoolExecutor(workers,initializer=_init_worker,initargs=(G,cfg,hop,bundle)) as pool: rows=list(pool.map(_center,centers,chunksize=cfg["job2_center_chunk_size"]))
    else:
        _init_worker(G,cfg,hop,bundle); rows=[_center(n) for n in centers]
    raw=pd.DataFrame(rows); raw.insert(0,"bundle",bundle); raw.insert(0,"hop",hop); raw.insert(0,"sim_idx",sim); raw.insert(0,"angle",angle); summary=[]
    for prop in [c for c in raw if c not in ("angle","sim_idx","hop","bundle","center_node")]: summary.append(dict(angle=angle,sim_idx=sim,hop=hop,bundle=bundle,property=prop,**distribution_summary(raw[prop])))
    atomic_csv(out,raw); atomic_csv(out.with_name(out.stem+"_summary.csv"),pd.DataFrame(summary)); mark_complete(out,cfg,{"angle":angle,"sim_idx":sim,"hop":hop,"bundle":bundle,"centers":len(centers)}); log.info("Complete %s",out)

if __name__=="__main__":main()
