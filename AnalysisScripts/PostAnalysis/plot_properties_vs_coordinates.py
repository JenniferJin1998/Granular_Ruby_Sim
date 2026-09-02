#!/usr/bin/env python3
"""Plot non-coordinate node/edge properties against x, y, and z.

Curves are calculated within each simulation first. Lines show the mean of
simulation binned means and bands show the simulation-level 95% t interval.
Thus particles/contacts are not treated as independent replicates.
"""
from __future__ import annotations
import argparse,re
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_INPUT=ROOT/"AnalysisResults"/"PeriodicBoudaries"/"2026-08-03"/"GraphPipeline"
DEFAULT_OUTPUT=ROOT/"AnalysisResults"/"PeriodicBoudaries"/"2026-08-03"/"AngleComparison"/"property_vs_coordinates"
ANGLES=("0deg","30deg");COLORS={"0deg":"#2474B5","30deg":"#D95F02"}

def safe(s):return re.sub(r"[^A-Za-z0-9_.-]+","_",str(s))
def numeric_properties(df,excluded):
    return [c for c in df if c not in excluded and not c.endswith("_with_walls") and pd.to_numeric(df[c],errors="coerce").notna().any()]
def simulation_profiles(df,coord,prop,edges):
    rows=[]
    for (angle,sim),g in df.groupby(["geometry","sim_idx"]):
        x=pd.to_numeric(g[coord],errors="coerce").to_numpy(float);y=pd.to_numeric(g[prop],errors="coerce").to_numpy(float);valid=np.isfinite(x)&np.isfinite(y);idx=np.digitize(x[valid],edges)-1
        for b in range(len(edges)-1):
            vals=y[valid][idx==b]
            if len(vals):rows.append((angle,int(sim),b,float(vals.mean())))
    return pd.DataFrame(rows,columns=["geometry","sim_idx","bin","value"])
def plot_level(frame,level,coord_map,props,out,bins):
    target=out/level;target.mkdir(parents=True,exist_ok=True);summary=[]
    for prop in props:
        fig,axes=plt.subplots(1,3,figsize=(15,4.4),sharey=True)
        all_y=[]
        for ax,(label,coord) in zip(axes,coord_map.items()):
            x=pd.to_numeric(frame[coord],errors="coerce");lo,hi=np.nanpercentile(x,[.25,99.75]);edges=np.linspace(lo,hi,bins+1);centers=(edges[:-1]+edges[1:])/2;profiles=simulation_profiles(frame,coord,prop,edges)
            for angle in ANGLES:
                pivot=profiles[profiles.geometry==angle].pivot(index="sim_idx",columns="bin",values="value").reindex(columns=range(bins));n=pivot.notna().sum().to_numpy();mean=pivot.mean().to_numpy();sd=pivot.std(ddof=1).to_numpy();ci=stats.t.ppf(.975,np.maximum(n-1,1))*sd/np.sqrt(np.maximum(n,1));ax.plot(centers,mean,color=COLORS[angle],label=angle);ax.fill_between(centers,mean-ci,mean+ci,color=COLORS[angle],alpha=.18);all_y.extend((mean-ci)[np.isfinite(mean-ci)]);all_y.extend((mean+ci)[np.isfinite(mean+ci)])
                for b,(m,c,nn) in enumerate(zip(mean,ci,n)):summary.append(dict(entity=level,property=prop,coordinate=label,geometry=angle,bin_center=centers[b],simulation_mean=m,ci95=c,n_simulations=nn))
            ax.set(xlabel=label,title=f"{prop} vs {label}");ax.legend(frameon=False)
        if all_y:
            qlo,qhi=np.percentile(all_y,[.5,99.5]);span=qhi-qlo
            if span>0:
                for ax in axes:ax.set_ylim(qlo-.05*span,qhi+.05*span)
        axes[0].set_ylabel(f"Mean {prop} per simulation bin");fig.suptitle(f"{level.capitalize()} property: {prop} (95% CI across simulations)");fig.tight_layout();fig.savefig(target/f"{safe(prop)}_vs_xyz.png",dpi=190);plt.close(fig)
    return summary
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT);ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);ap.add_argument("--bins",type=int,default=30);args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);all_rows=[]
    node=pd.read_csv(args.input_dir/"node_features.csv",low_memory=False);node=node[node.geometry.isin(ANGLES)];node_ex={"geometry","sim_idx","node_id","x","y","z","is_wall"};node_props=numeric_properties(node,node_ex);all_rows+=plot_level(node,"node",{"x":"x","y":"y","z":"z"},node_props,args.output_dir,args.bins)
    edge=pd.read_csv(args.input_dir/"edge_features.csv",low_memory=False);edge=edge[edge.geometry.isin(ANGLES)];edge_ex={"geometry","sim_idx","node1","node2","contact_x","contact_y","contact_z"};edge_props=numeric_properties(edge,edge_ex);all_rows+=plot_level(edge,"edge",{"x":"contact_x","y":"contact_y","z":"contact_z"},edge_props,args.output_dir,args.bins)
    pd.DataFrame(all_rows).to_csv(args.output_dir/"property_coordinate_profiles.csv",index=False);print(f"Saved {len(node_props)} node and {len(edge_props)} edge property profiles to {args.output_dir}")
if __name__=="__main__":main()
