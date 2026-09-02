#!/usr/bin/env python3
"""Visualize merged Job 2 local-subgraph results using simulations as replicates."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pipeline_common import CONFIG_PATH,load_config,ensure_layout,finite,geometry_colors

ANGLES=("0deg","30deg");COLORS={"0deg":"#2474B5","30deg":"#D95F02"};HOPS=(2,3,4,5)
DISPLAY_STATS=("mean","std","iqr","fwhm","skewness","kurtosis")
def safe(x):return re.sub(r"[^A-Za-z0-9_.-]+","_",str(x))
def ci95(x):
    x=finite(x)
    return stats.t.ppf(.975,len(x)-1)*stats.sem(x) if len(x)>1 else np.nan
def scale_axis(ax,values):
    x=finite(values)
    if len(x)<2:return
    lo,hi=np.percentile(x,[.5,99.5]);span=hi-lo
    if span>0:ax.set_ylim(lo-.08*span,hi+.08*span)
    nz=np.abs(x[np.abs(x)>np.finfo(float).eps])
    if len(nz) and nz.max()/nz.min()>1000:
        if np.all(x>0):ax.set_yscale("log")
        else:ax.set_yscale("symlog",linthresh=max(np.percentile(nz,10),np.finfo(float).eps))
def plot_property_points(data,prop,out):
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharey=True);all_values=[]
    for ax,hop in zip(axes.flat,HOPS):
        for xpos,angle in enumerate(ANGLES):
            vals=finite(data[(data.property==prop)&(data.hop==hop)&(data.angle==angle)]["mean"]);jitter=np.linspace(-.07,.07,len(vals)) if len(vals)>1 else np.zeros(len(vals));ax.scatter(xpos+jitter,vals,s=24,alpha=.7,color=COLORS[angle]);ax.errorbar(xpos,np.mean(vals),yerr=ci95(vals),fmt="D",color="black",capsize=4,zorder=5);all_values.extend(vals)
        ax.set_xticks([0,1],["0°","30°"]);ax.set_title(f"{hop}-hop")
    scale_axis(axes[0,0],all_values)
    if axes[0,0].get_yscale()!="linear":
        for ax in axes.flat:ax.set_yscale(axes[0,0].get_yscale())
    fig.suptitle(f"Local {prop}: simulation means and 95% CI");fig.supylabel(f"Simulation mean {prop}");fig.tight_layout();fig.savefig(out/"angle_points"/f"{safe(prop)}.png",dpi=200);plt.close(fig)
def plot_property_statistics(data,prop,out):
    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    for ax,stat_name in zip(axes.flat,DISPLAY_STATS):
        all_values=[]
        for angle in ANGLES:
            means=[];cis=[]
            for hop in HOPS:
                vals=finite(data[(data.property==prop)&(data.hop==hop)&(data.angle==angle)][stat_name]);means.append(np.mean(vals) if len(vals) else np.nan);cis.append(ci95(vals));all_values.extend(vals)
            ax.errorbar(HOPS,means,yerr=cis,marker="o",capsize=3,color=COLORS[angle],label=angle)
        scale_axis(ax,all_values);ax.set(title=stat_name.replace("_"," "),xlabel="Hop size",xticks=HOPS);ax.legend(frameon=False)
    fig.suptitle(f"Local-distribution summaries across simulations: {prop}");fig.tight_layout();fig.savefig(out/"statistic_trends"/f"{safe(prop)}.png",dpi=200);plt.close(fig)
def heatmaps(tests,out):
    props=sorted(tests.property.unique())
    for (geometry_a,geometry_b),subset in tests.groupby(["geometry_a","geometry_b"]):
        fig,axes=plt.subplots(1,2,figsize=(10,max(6,.28*len(props))))
        for ax,column,title,cmap,vmin,vmax in ((axes[0],"effect_size",f"Hedges g ({geometry_b} − {geometry_a})","coolwarm",-3,3),(axes[1],"test_p","−log10 Welch p","magma",0,None)):
            matrix=subset.pivot(index="property",columns="hop",values=column).reindex(index=props,columns=HOPS);values=matrix.to_numpy(float)
            if column=="test_p":values=-np.log10(np.clip(values,1e-300,1))
            im=ax.imshow(values,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax);ax.set_xticks(range(len(HOPS)),HOPS);ax.set_yticks(range(len(props)),props if ax is axes[0] else [""]*len(props));ax.set(xlabel="Hop size",title=title);fig.colorbar(im,ax=ax,shrink=.75)
        fig.tight_layout();fig.savefig(out/f"job2_{safe(geometry_a)}_vs_{safe(geometry_b)}_heatmaps.png",dpi=220);plt.close(fig)
def local_complete(frame,out):
    if frame is None or frame.empty:return
    for prop,g in frame.groupby("property"):
        fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=False,sharey=True)
        values=[]
        for ax,hop in zip(axes,HOPS):
            h=g[g.hop==hop]
            for angle in ANGLES:
                a=h[h.angle==angle];ax.scatter(a.complete_graph_value,a.local_mean,s=22,alpha=.7,color=COLORS[angle],label=angle);values.extend(a.complete_graph_value);values.extend(a.local_mean)
            finite_v=finite(values);lo,hi=np.min(finite_v),np.max(finite_v);ax.plot([lo,hi],[lo,hi],"--",color="0.4",lw=1);ax.set(title=f"{hop}-hop",xlabel="Complete graph")
        axes[0].set_ylabel("Mean local value");axes[-1].legend(frameon=False);fig.suptitle(f"Local versus complete: {prop}");fig.tight_layout();fig.savefig(out/"local_vs_complete"/f"{safe(prop)}.png",dpi=200);plt.close(fig)
def main():
    global ANGLES,COLORS,HOPS
    ap=argparse.ArgumentParser();ap.add_argument("--config",default=CONFIG_PATH);args=ap.parse_args();cfg=load_config(args.config);ANGLES=tuple(cfg["angles"]);COLORS=geometry_colors(cfg);HOPS=tuple(cfg["hop_sizes"]);root=ensure_layout(cfg);combined=root/"combined_results";out=combined/"job2_plots"
    for name in ("angle_points","statistic_trends","local_vs_complete"):(out/name).mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(combined/"local_subgraph_summaries.csv");tests=pd.read_csv(combined/"local_angle_tests.csv");properties=sorted(data.property.unique())
    for prop in properties:plot_property_points(data,prop,out);plot_property_statistics(data,prop,out)
    heatmaps(tests,out);path=combined/"local_vs_complete_graph.csv";local_complete(pd.read_csv(path) if path.exists() else None,out)
    print(f"Saved Job 2 plots for {len(properties)} properties to {out}")
if __name__=="__main__":main()
