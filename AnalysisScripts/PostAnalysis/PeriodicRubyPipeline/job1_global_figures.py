#!/usr/bin/env python3
"""Job 1: particle-topology evidence with simulation-level inference."""
import argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pipeline_common import *


def hedges(a,b):
    if len(a)<2 or len(b)<2:return np.nan
    df=len(a)+len(b)-2; pv=((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/df
    return ((np.mean(b)-np.mean(a))/np.sqrt(pv))*(1-3/(4*df-1)) if pv>0 else 0.0


def bh_adjust(p):
    p=np.asarray(p,float); out=np.full(len(p),np.nan); valid=np.where(np.isfinite(p))[0]
    if not len(valid):return out
    order=valid[np.argsort(p[valid])]; ranked=p[order]*len(order)/np.arange(1,len(order)+1); ranked=np.minimum.accumulate(ranked[::-1])[::-1]; out[order]=np.minimum(ranked,1); return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config); root=ensure_layout(cfg); log=setup_logging(cfg,"job1")
    if args.dry_run: print("Would reuse saved core properties and compute missing triangle counts/figures"); return
    graphs=load_graph_dict(cfg); rows=[]; distributions=[]
    loop_rows=[]
    for angle,sim in graph_index(graphs,cfg):
        G=graphs[angle]["core"][sim]; n=G.number_of_nodes(); triangles=nx.triangles(G); loops=G.graph.get("loop_counts",{}) or {}
        degree=np.asarray([d for _,d in G.degree()],float); clustering=np.asarray(list(nx.clustering(G).values()),float)
        rows.append(dict(angle=angle,sim_idx=sim,mean_degree=degree.mean(),mean_closeness=np.mean([d.get("closeness",np.nan) for _,d in G.nodes(data=True)]),
                         mean_clustering=clustering.mean(),triangles_per_node=sum(triangles.values())/(3*n),four_loops_per_node=float(loops.get(4,loops.get("4",0)))/n))
        for prop,vals in (("degree",degree),("local_clustering",clustering),("node_triangles",np.asarray(list(triangles.values()),float))):
            for v in vals: distributions.append(dict(angle=angle,sim_idx=sim,property=prop,value=v))
        total=sum(loops.values())
        for size,count in loops.items():
            distributions.append(dict(angle=angle,sim_idx=sim,property="normalized_loop_size",value=int(size),weight=count/total if total else 0))
            loop_rows.append(dict(angle=angle,sim_idx=sim,loop_size=int(size),fraction=count/total if total else 0))
    frame=pd.DataFrame(rows)
    tests=[]
    for prop in [c for c in frame if c not in ("angle","sim_idx")]:
        pair_tests=pairwise_replicate_tests(frame[["angle",prop]].rename(columns={prop:"value"}),"value",[],cfg)
        if len(pair_tests): pair_tests.insert(0,"property",prop); tests.append(pair_tests)
    tests=pd.concat(tests,ignore_index=True) if tests else pd.DataFrame()
    if len(tests): tests["test_p_fdr_bh"]=bh_adjust(tests.test_p)
    out=root/"job1_global_figures"; atomic_csv(out/"simulation_topology.csv",frame); atomic_csv(out/"topology_tests.csv",tests); atomic_csv(out/"node_distributions.csv",pd.DataFrame(distributions))
    colors=geometry_colors(cfg)
    metrics=["mean_degree","mean_closeness","triangles_per_node","four_loops_per_node"]; fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,prop in zip(axes.flat,metrics):
        for x,angle in enumerate(cfg["angles"]):
            color=colors[angle]
            vals=frame.loc[frame.angle==angle,prop].to_numpy(); ax.scatter(np.full(len(vals),x),vals,color=color,alpha=.7); ax.errorbar(x,vals.mean(),yerr=stats.t.ppf(.975,len(vals)-1)*stats.sem(vals),fmt="ks",capsize=4)
        ax.set_xticks(range(len(cfg["angles"])),cfg["angles"]); ax.set_title(prop.replace("_"," "))
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(out/f"figure1_statistical_evidence.{ext}",dpi=250)
    plt.close(fig)
    dist=pd.DataFrame(distributions); loops_df=pd.DataFrame(loop_rows)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for angle in cfg["angles"]:
        color=colors[angle]
        subset=loops_df[loops_df.angle==angle].groupby("loop_size").fraction.mean(); axes[0,0].plot(subset.index,subset.values,"o-",label=angle,color=color)
        for ax,prop in ((axes[0,1],"degree"),(axes[1,0],"local_clustering")):
            vals=dist[(dist.angle==angle)&(dist.property==prop)].value; ax.hist(vals,bins=40,density=True,histtype="step",label=angle,color=color)
    axes[0,0].set(title="Normalized loop-size distribution",xlabel="Loop size",ylabel="Mean fraction"); axes[0,1].set(title="Node-degree distribution",xlabel="Degree",ylabel="Density"); axes[1,0].set(title="Local-clustering distribution",xlabel="Clustering",ylabel="Density")
    for angle in cfg["angles"]:
        color=colors[angle]
        G=graphs[angle]["core"][0]; center=max(G.degree,key=lambda x:x[1])[0]; H=G.subgraph(nx.single_source_shortest_path_length(G,center,cutoff=2)); vals=[d for _,d in H.degree()]; axes[1,1].scatter(range(len(vals)),sorted(vals),s=12,label=angle,color=color,alpha=.7)
    axes[1,1].set(title="Representative 2-hop neighborhoods",xlabel="Ranked particle",ylabel="Degree")
    for ax in axes.flat: ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(out/f"figure2_structural_interpretation.{ext}",dpi=250)
    plt.close(fig)
    ncols=min(2,len(cfg["angles"]));nrows=int(np.ceil(len(cfg["angles"])/ncols));fig=plt.figure(figsize=(6*ncols,5*nrows))
    for panel,angle in enumerate(cfg["angles"],1):
        color=colors[angle];ax=fig.add_subplot(nrows,ncols,panel,projection="3d"); G=graphs[angle]["core"][0]; center=max(G.degree,key=lambda x:x[1])[0]; H=G.subgraph(nx.single_source_shortest_path_length(G,center,cutoff=3)); pos={n:G.nodes[n]["position"] for n in H}
        for u,v in H.edges(): ax.plot(*zip(pos[u],pos[v]),color="0.75",lw=.5,alpha=.5)
        xyz=np.asarray(list(pos.values())); ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=8,color=color); ax.set_title(f"{angle} representative 3-hop particle graph")
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(out/f"representative_3d_graphs.{ext}",dpi=250)
    plt.close(fig); mark_complete(out/"simulation_topology.csv",cfg); log.info("Job 1 complete")

if __name__=="__main__":main()
