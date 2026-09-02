#!/usr/bin/env python3
"""Create simple four-view system plots highlighting stored high-force labels."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pipeline_common import *

VIEWS=((0,0,"View along x"),(0,90,"View along y"),(90,-90,"View along z"),(24,-52,"Perspective 3D"))


def split_periodic_segment(start,delta,box_lengths,periodic_axes):
    """Split a minimum-image edge where it crosses an orthogonal periodic face."""
    current=np.asarray(start,float).copy(); remaining=np.asarray(delta,float).copy(); pieces=[]
    for _ in range(len(periodic_axes)+1):
        endpoint=current+remaining; crossings=[]
        for axis in periodic_axes:
            length=box_lengths[axis]
            if endpoint[axis]<0 and remaining[axis]<0: crossings.append(((0-current[axis])/remaining[axis],axis,0.0,length))
            elif endpoint[axis]>length and remaining[axis]>0: crossings.append(((length-current[axis])/remaining[axis],axis,length,0.0))
        crossings=[x for x in crossings if 0<x[0]<1]
        if not crossings: pieces.append(np.vstack((current,endpoint)));break
        t=min(x[0] for x in crossings);hit=current+t*remaining;pieces.append(np.vstack((current,hit)));remaining=(1-t)*remaining;current=hit.copy()
        for cross_t,axis,_,wrapped in crossings:
            if np.isclose(cross_t,t):current[axis]=wrapped
    return pieces


def graph_geometry(root):
    geo=json.loads((root/"job0_metadata"/"geometry_estimate.json").read_text());return {int(k):float(v) for k,v in geo["box_lengths"].items()},[int(x) for x in geo["periodic_axes"]]


def plot_system_on_axes(axes,G,cfg,box_lengths,periodic_axes,angle,sim):
    nodes=list(G.nodes());xyz=np.asarray([G.nodes[n]["position"] for n in nodes],float);index={n:i for i,n in enumerate(nodes)};node_labels,edge_labels,_=high_force_labels(G,cfg)
    gray_segments=[];red_segments=[]
    for u,v in G.edges():
        start=xyz[index[u]];delta=minimum_image(xyz[index[v]]-start,box_lengths,periodic_axes);target=red_segments if edge_labels[(u,v)] else gray_segments;target.extend(split_periodic_segment(start,delta,box_lengths,periodic_axes))
    high=np.asarray([node_labels[n] for n in nodes],bool);mins=xyz.min(0);maxs=xyz.max(0);span=np.maximum(maxs-mins,np.finfo(float).eps)
    for ax,(elev,azim,title) in zip(axes,VIEWS):
        ax.add_collection3d(Line3DCollection(gray_segments,colors="#888888",linewidths=.28,alpha=.20,rasterized=True));ax.add_collection3d(Line3DCollection(red_segments,colors="#D62728",linewidths=1.15,alpha=.92,rasterized=True))
        ax.scatter(xyz[~high,0],xyz[~high,1],xyz[~high,2],s=2.2,c="#9A9A9A",alpha=.42,depthshade=False,rasterized=True);ax.scatter(xyz[high,0],xyz[high,1],xyz[high,2],s=8,c="#D62728",alpha=.95,depthshade=False,rasterized=True)
        ax.view_init(elev=elev,azim=azim);ax.set_proj_type("persp" if title=="Perspective 3D" else "ortho");ax.set(xlim=(mins[0],maxs[0]),ylim=(mins[1],maxs[1]),zlim=(mins[2],maxs[2]),title=title);ax.set_box_aspect(span);ax.set_axis_off()
    axes[0].text2D(.02,.96,f"{angle}, simulation {sim}",transform=axes[0].transAxes,fontsize=11,weight="bold")
    return len(red_segments),int(high.sum())


def generate_views(cfg,root):
    out=root/"combined_results"/"basic_system_views";out.mkdir(parents=True,exist_ok=True);graphs=load_graph_dict(cfg);box_lengths,periodic_axes=graph_geometry(root);rng=np.random.default_rng(cfg["random_seed"]+56);selected={angle:int(rng.integers(0,len(graphs[angle][cfg["primary_graph_view"]]))) for angle in cfg["angles"]};records=[]
    combined=plt.figure(figsize=(18,8));combined_axes=[]
    for row,angle in enumerate(cfg["angles"]):
        sim=selected[angle];G=graphs[angle][cfg["primary_graph_view"]][sim];axes=[combined.add_subplot(len(cfg["angles"]),4,row*4+i+1,projection="3d") for i in range(4)];red_edges,red_nodes=plot_system_on_axes(axes,G,cfg,box_lengths,periodic_axes,angle,sim);combined_axes.extend(axes);records.append(dict(angle=angle,sim_idx=sim,high_force_nodes=red_nodes,high_force_edges=sum(truthy_label(d.get(cfg["high_force_edge_label"],False)) for *_,d in G.edges(data=True)),node_label=cfg["high_force_node_label"],edge_label=cfg["high_force_edge_label"]))
        fig=plt.figure(figsize=(12,10));single_axes=[fig.add_subplot(2,2,i+1,projection="3d") for i in range(4)];plot_system_on_axes(single_axes,G,cfg,box_lengths,periodic_axes,angle,sim);fig.legend(handles=[Line2D([0],[0],marker="o",linestyle="",color="#D62728",label="High-force node"),Line2D([0],[0],color="#D62728",lw=1.5,label="High-force edge"),Line2D([0],[0],marker="o",linestyle="",color="#888888",label="Other node/contact")],loc="lower center",ncol=3,frameon=False);fig.subplots_adjust(bottom=.08,wspace=.02,hspace=.05);fig.savefig(out/f"{angle}_sim{sim:03d}_four_views.png",dpi=240,bbox_inches="tight");plt.close(fig)
    combined.legend(handles=[Line2D([0],[0],marker="o",linestyle="",color="#D62728",label="High-force node"),Line2D([0],[0],color="#D62728",lw=1.5,label="High-force edge"),Line2D([0],[0],marker="o",linestyle="",color="#888888",label="Other node/contact")],loc="lower center",ncol=3,frameon=False);combined.subplots_adjust(bottom=.07,wspace=.03,hspace=.05);combined.savefig(out/"selected_geometries_four_views.png",dpi=240,bbox_inches="tight");plt.close(combined);atomic_csv(out/"selected_simulations.csv",pd.DataFrame(records));return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default=CONFIG_PATH);args=ap.parse_args();cfg=load_config(args.config);root=ensure_layout(cfg);print(generate_views(cfg,root))


if __name__=="__main__":main()
