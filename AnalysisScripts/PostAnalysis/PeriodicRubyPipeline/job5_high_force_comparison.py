#!/usr/bin/env python3
"""Job 5: high-force/non-high-force complete-graph and centered-subgraph comparisons."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline_common import *

ID_COLUMNS={"angle","sim_idx","load_step","hop","center_node","center_group","node_id","edge_u","edge_v","group"}


def tasks(graphs,cfg):
    indexed=graph_index(graphs,cfg)
    return [("complete",a,s,None) for a,s in indexed]+[("centered",a,s,h) for a,s in indexed for h in cfg["hop_sizes"]]


def aggregate(values,percentiles):
    x=finite(values)
    out={"mean":np.nan,"median":np.nan,"std":np.nan,"iqr":np.nan}
    out.update({f"p{p:02d}":np.nan for p in percentiles})
    if not len(x): return out
    out.update(mean=float(x.mean()),median=float(np.median(x)),std=float(x.std(ddof=1)) if len(x)>1 else np.nan,iqr=float(np.percentile(x,75)-np.percentile(x,25)))
    out.update({f"p{p:02d}":float(np.percentile(x,p)) for p in percentiles})
    return out


def summarize_groups(frame,id_columns):
    rows=[]
    for prop in [c for c in frame if c not in id_columns]:
        for group,g in frame.groupby("group" if "group" in frame else "center_group"):
            rows.append({"property":prop,"group":group,**distribution_summary(g[prop])})
    return pd.DataFrame(rows)


def complete_task(G,cfg,angle,sim,out):
    node_labels,edge_labels,label_source=high_force_labels(G,cfg)
    incident={n for edge,flag in edge_labels.items() if flag for n in edge}; stored_high={n for n,flag in node_labels.items() if flag}; label_extra=len(stored_high-incident); label_missing=len(incident-stored_high)
    node_records=[d for _,d in G.nodes(data=True)]; edge_records=[d for *_,d in G.edges(data=True)]
    node_props=scalar_numeric_properties(node_records,cfg,excluded={cfg["high_force_node_label"],"is_wall"})
    edge_props=scalar_numeric_properties(edge_records,cfg,excluded={cfg["high_force_edge_label"],"is_wall_contact"})
    nodes=[]
    for n,d in G.nodes(data=True):
        row={"angle":angle,"sim_idx":sim,"load_step":cfg["load_step"],"node_id":n,"group":"high_force" if node_labels[n] else "non_high_force"}
        row.update({p:d.get(p,np.nan) for p in node_props}); nodes.append(row)
    edges=[]
    for u,v,d in G.edges(data=True):
        row={"angle":angle,"sim_idx":sim,"load_step":cfg["load_step"],"edge_u":u,"edge_v":v,"group":"high_force" if edge_labels[(u,v)] else "non_high_force"}
        row.update({p:d.get(p,np.nan) for p in edge_props}); edges.append(row)
    ndf=pd.DataFrame(nodes); edf=pd.DataFrame(edges)
    ns=summarize_groups(ndf,ID_COLUMNS); ns.insert(0,"scope","node")
    es=summarize_groups(edf,ID_COLUMNS); es.insert(0,"scope","edge")
    summary=pd.concat([ns,es],ignore_index=True); summary.insert(0,"node_label_missing_edge_endpoints",label_missing);summary.insert(0,"node_label_extra_vs_edge_endpoints",label_extra);summary.insert(0,"label_source",label_source); summary.insert(0,"load_step",cfg["load_step"]); summary.insert(0,"sim_idx",sim); summary.insert(0,"angle",angle)
    base=out.name.removesuffix("_summary.csv"); atomic_csv(out.with_name(base+"_nodes.csv"),ndf); atomic_csv(out.with_name(base+"_edges.csv"),edf); atomic_csv(out,summary)
    mark_complete(out,cfg,{"mode":"complete","angle":angle,"sim_idx":sim,"node_label_source":label_source,"node_label_extra_vs_edge_endpoints":label_extra,"node_label_missing_edge_endpoints":label_missing,"node_properties":len(node_props),"edge_properties":len(edge_props)})


def job2_metrics(root,angle,sim,hop,cfg):
    merged=None
    keys=["center_node"]
    for bundle in cfg["job2_property_bundles"]:
        path=root/"job2_subgraphs"/f"{angle}_sim{sim:03d}_hop{hop}_{bundle}.csv"
        if not path.exists(): raise FileNotFoundError(f"Job 2 prerequisite missing: {path}")
        frame=pd.read_csv(path).drop(columns=["angle","sim_idx","hop","bundle"],errors="ignore")
        if merged is None: merged=frame
        else:
            new=[c for c in frame if c not in merged or c in keys]
            merged=merged.merge(frame[new],on=keys,how="outer",validate="one_to_one")
    return merged.set_index("center_node").to_dict("index")


def centered_task(G,cfg,root,angle,sim,hop,out):
    node_labels,_,label_source=high_force_labels(G,cfg)
    node_props=scalar_numeric_properties([d for _,d in G.nodes(data=True)],cfg,excluded={cfg["high_force_node_label"],"is_wall"})
    edge_props=scalar_numeric_properties([d for *_,d in G.edges(data=True)],cfg,excluded={cfg["high_force_edge_label"],"is_wall_contact"})
    topology=job2_metrics(root,angle,sim,hop,cfg); percentiles=[int(x) for x in cfg["job5_selected_percentiles"]]; rows=[]
    for center in G.nodes():
        members=nx.single_source_shortest_path_length(G,center,cutoff=hop).keys(); H=G.subgraph(members); cd=G.nodes[center]
        row={"angle":angle,"sim_idx":sim,"load_step":cfg["load_step"],"hop":hop,"center_node":center,"center_group":"high_force" if node_labels[center] else "non_high_force"}
        for prop in node_props: row[f"center__{prop}"]=cd.get(prop,np.nan)
        metrics=topology.get(center,topology.get(str(center),{}))
        for prop,value in metrics.items(): row[f"graph__{prop}"]=value
        for scope,props,records in (("node",node_props,[d for _,d in H.nodes(data=True)]),("edge",edge_props,[d for *_,d in H.edges(data=True)])):
            for prop in props:
                values=[d.get(prop,np.nan) for d in records]
                for stat,value in aggregate(values,percentiles).items(): row[f"{scope}_{stat}__{prop}"]=value
        rows.append(row)
    raw=pd.DataFrame(rows); summary=summarize_groups(raw,ID_COLUMNS); summary.insert(0,"label_source",label_source); summary.insert(0,"load_step",cfg["load_step"]); summary.insert(0,"hop",hop); summary.insert(0,"sim_idx",sim); summary.insert(0,"angle",angle)
    base=out.name.removesuffix("_summary.csv"); atomic_csv(out.with_name(base+"_subgraphs.csv"),raw); atomic_csv(out,summary)
    mark_complete(out,cfg,{"mode":"centered","angle":angle,"sim_idx":sim,"hop":hop,"centers":len(raw),"node_label_source":label_source})


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--task-id",type=int); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
    cfg=load_config(args.config); root=ensure_layout(cfg); graphs=load_graph_dict(cfg); all_tasks=tasks(graphs,cfg); tid=task_id(args.task_id)
    if tid is None: raise SystemExit("Provide --task-id or SLURM_ARRAY_TASK_ID")
    if tid<0 or tid>=len(all_tasks): raise SystemExit(f"task id must be 0..{len(all_tasks)-1}")
    mode,angle,sim,hop=all_tasks[tid]; stem=f"{angle}_sim{sim:03d}_{mode}"+(f"_hop{hop}" if hop else ""); out=root/"job5_high_force_comparison"/f"{stem}_summary.csv"; log=setup_logging(cfg,"job5",tid)
    if output_valid(out,cfg,("angle","sim_idx","scope","property","group") if mode=="complete" else ("angle","sim_idx","hop","property","group")): log.info("Skipping valid %s",out); return
    if args.dry_run: print(tid,mode,angle,sim,hop,out); return
    G=graphs[angle][cfg["primary_graph_view"]][sim]
    if mode=="complete": complete_task(G,cfg,angle,sim,out)
    else: centered_task(G,cfg,root,angle,sim,hop,out)
    log.info("Complete %s",out)


if __name__=="__main__": main()
