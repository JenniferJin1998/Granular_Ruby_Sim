#!/usr/bin/env python3
"""Job 3: ideal-crystal complete and local graph baselines."""
import argparse
import pandas as pd
from pipeline_common import *
from crystal_common import STRUCTURES,crystal_graph

def tasks(cfg): return [(s,h) for s in STRUCTURES for h in cfg["hop_sizes"]]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--task-id",type=int); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config); root=ensure_layout(cfg); all_tasks=tasks(cfg); tid=task_id(args.task_id)
    if tid is None: raise SystemExit("Provide --task-id or SLURM_ARRAY_TASK_ID")
    structure,hop=all_tasks[tid]; out=root/"job3_crystal_baselines"/f"{structure}_hop{hop}.csv"; log=setup_logging(cfg,"job3",tid)
    if output_valid(out,cfg,("center_node","mean_degree")): log.info("Skipping %s",out); return
    if args.dry_run: print(tid,structure,hop,out); return
    G=crystal_graph(structure,cfg["particle_diameter"],cfg["contact_distance_tolerance_fraction"])
    # Ideal structures are periodic laterally. Particles in the same z plane
    # with the same coordination are symmetry-equivalent, so calculate each
    # environment once and expand it back to one row per particle.
    groups={}
    for center,d in G.nodes(data=True):
        key=(round(float(d["position"][2]),12),int(G.degree(center))); groups.setdefault(key,[]).append(center)
    rows=[]
    for equivalent_centers in groups.values():
        representative=equivalent_centers[0]; nodes=nx.single_source_shortest_path_length(G,representative,cutoff=hop).keys(); metrics=safe_graph_metrics(G.subgraph(nodes),cfg,"all")
        for center in equivalent_centers: rows.append(dict(center_node=center,**metrics))
    raw=pd.DataFrame(rows); raw.insert(0,"hop",hop); raw.insert(0,"structure",structure); summary=[dict(structure=structure,hop=hop,property=p,**distribution_summary(raw[p])) for p in raw if p not in ("structure","hop","center_node")]
    atomic_csv(out,raw); atomic_csv(out.with_name(out.stem+"_summary.csv"),pd.DataFrame(summary))
    meta={k:v for k,v in G.graph.items()}; meta.update(nodes=G.number_of_nodes(),edges=G.number_of_edges(),coordination_mean=2*G.number_of_edges()/G.number_of_nodes(),symmetry_environment_groups=len(groups))
    atomic_json(out.with_name(out.stem+"_metadata.json"),meta); mark_complete(out,cfg,meta); log.info("Complete %s",out)
if __name__=="__main__":main()
