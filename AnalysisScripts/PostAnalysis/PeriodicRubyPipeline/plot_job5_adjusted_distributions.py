#!/usr/bin/env python3
"""Create optional adjusted Job 5 closeness and degree plots without overwriting originals."""
import argparse
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pipeline_common import CONFIG_PATH,ensure_layout,finite,load_config

COLORS={"non_high_force":"#2474B5","high_force":"#D95F02"}
LABELS={"non_high_force":"non high force","high_force":"high force"}


def load_nodes(root):
    paths=sorted((root/"job5_high_force_comparison").glob("*_complete_nodes.csv"))
    if not paths:raise FileNotFoundError("No Job 5 complete-node tables found")
    return pd.concat([pd.read_csv(p,usecols=["angle","group","closeness","degree"]) for p in paths],ignore_index=True)


def break_marks(left,right):
    kwargs=dict(color="black",clip_on=False,lw=1)
    left.plot((1-.012,1+.012),(-.018,+.018),transform=left.transAxes,**kwargs);left.plot((1-.012,1+.012),(1-.018,1+.018),transform=left.transAxes,**kwargs)
    right.plot((-.012,+.012),(-.018,+.018),transform=right.transAxes,**kwargs);right.plot((-.012,+.012),(1-.018,1+.018),transform=right.transAxes,**kwargs)


def plot_closeness(frame,path):
    fig=plt.figure(figsize=(12,4.8));grid=GridSpec(1,5,figure=fig,width_ratios=[1.15,4,.6,1.15,4],wspace=.06);axes=[fig.add_subplot(grid[i]) for i in (0,1,3,4)];bins=np.linspace(0,.12,61);maximum_probability=0
    for panel,angle in enumerate(("0deg","30deg")):
        left,right=axes[2*panel:2*panel+2]
        for group in ("non_high_force","high_force"):
            values=finite(frame.loc[(frame.angle==angle)&(frame.group==group),"closeness"]);weights=np.ones(len(values))/len(values) if len(values) else None;probability,edges=np.histogram(values,bins=bins,weights=weights)
            maximum_probability=max(maximum_probability,float(probability.max()*100))
            for ax in (left,right):ax.stairs(probability*100,edges,label=LABELS[group],color=COLORS[group],lw=1.6)
        left.set_xlim(-.0003,.004);right.set_xlim(.08,.12);left.spines["right"].set_visible(False);right.spines["left"].set_visible(False);right.tick_params(axis="y",left=False,labelleft=False);left.set_yscale("log");right.set_yscale("log");break_marks(left,right);right.set_title(angle,fontsize=14);right.legend(frameon=False,loc="upper left")
        if panel==0:left.set_ylabel("Probability per 0.002-wide bin (%)")
        left.set_xlabel("zero");right.set_xlabel("closeness (0.08–0.12)")
    for ax in axes:ax.set_ylim(.001,maximum_probability*1.3)
    fig.suptitle("Node closeness: zero population and continuous range",fontsize=16);fig.subplots_adjust(top=.84,bottom=.16,left=.08,right=.98);fig.savefig(path,dpi=220);plt.close(fig)


def plot_degree(frame,path):
    fig=plt.figure(figsize=(11,7));grid=GridSpec(2,2,figure=fig,height_ratios=[2.4,1.25],hspace=.36);axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])];difference_ax=fig.add_subplot(grid[1,:]);degrees=np.arange(0,11);width=.38;probabilities={}
    for ax,angle in zip(axes,("0deg","30deg")):
        for offset,group in ((-width/2,"non_high_force"),(width/2,"high_force")):
            values=finite(frame.loc[(frame.angle==angle)&(frame.group==group),"degree"]).astype(int);counts=np.asarray([(values==degree).sum() for degree in degrees],float);probability=100*counts/counts.sum() if counts.sum() else counts;ax.bar(degrees+offset,probability,width=width,label=LABELS[group],color=COLORS[group],alpha=.82,edgecolor="white",linewidth=.4)
            probabilities[(angle,group)]=probability
        ax.set_xticks(degrees);ax.set(xlabel="Degree",title=angle,xlim=(-.7,10.7));ax.grid(axis="y",alpha=.2);ax.legend(frameon=False)
    top_max=max(float(values.max()) for values in probabilities.values());[ax.set_ylim(0,top_max*1.12) for ax in axes];axes[0].set_ylabel("Fraction within group (%)")
    for offset,group in ((-width/2,"non_high_force"),(width/2,"high_force")):
        difference=probabilities[("30deg",group)]-probabilities[("0deg",group)];difference_ax.bar(degrees+offset,difference,width=width,label=LABELS[group],color=COLORS[group],alpha=.82,edgecolor="white",linewidth=.4)
    difference_ax.axhline(0,color="black",lw=.8);difference_ax.set_xticks(degrees);difference_ax.set(xlabel="Degree",ylabel="30° − 0°\n(percentage points)",xlim=(-.7,10.7),title="Angle-dependent difference");difference_ax.grid(axis="y",alpha=.2);difference_ax.legend(frameon=False,ncol=2)
    fig.suptitle("Node degree distributions: exact integer values",fontsize=16);fig.subplots_adjust(top=.90,bottom=.09,left=.08,right=.98);fig.savefig(path,dpi=220);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default=CONFIG_PATH);args=ap.parse_args();cfg=load_config(args.config);root=ensure_layout(cfg);out=root/"combined_results"/"job5_high_force_plots"/"node_distributions";out.mkdir(parents=True,exist_ok=True);nodes=load_nodes(root);plot_closeness(nodes,out/"closeness_adjusted.png");plot_degree(nodes,out/"degree_adjusted.png");print(out)


if __name__=="__main__":main()
