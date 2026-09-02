#!/usr/bin/env python3
"""Merge completed task outputs and run simulation-replicate comparisons."""
import argparse, json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.spatial import distance as spatial_distance
from pipeline_common import *

JOB2_NAME_RE=re.compile(r"^(.+)_sim\d{3}_hop\d+_(fast|paths|loops|spectral|connectivity)(?:_summary)?\.csv$")
ID_COLUMNS={"angle","sim_idx","load_step","hop","center_node","center_group","node_id","edge_u","edge_v","group"}

def current_job2_files(directory,summary,cfg):
    files=[]
    for path in Path(directory).glob("*.csv"):
        if bool(path.name.endswith("_summary.csv"))!=bool(summary): continue
        match=JOB2_NAME_RE.match(path.name)
        if match and match.group(1) in cfg["angles"]: files.append(path)
    return sorted(files)

def hedges(a,b):
    if len(a)<2 or len(b)<2:return np.nan
    df=len(a)+len(b)-2; pv=((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/df
    return (np.mean(b)-np.mean(a))/np.sqrt(pv)*(1-3/(4*df-1)) if pv>0 else 0.0

def js_distance(a,b,bins=128):
    a=finite(a);b=finite(b)
    if not len(a) or not len(b):return np.nan
    lo=min(a.min(),b.min());hi=max(a.max(),b.max())
    if hi<=lo:return 0.0
    edges=np.linspace(lo,hi,bins+1);pa,_=np.histogram(a,edges);pb,_=np.histogram(b,edges)
    return float(spatial_distance.jensenshannon(pa+1e-12,pb+1e-12,base=2))

def plot_hist_safe(ax,values,label,color=None):
    x=finite(values)
    if not len(x): return np.array([])
    variable=np.ptp(x)>100*np.finfo(float).eps*max(1.0,float(np.max(np.abs(x))))
    if variable:
        density,edges=np.histogram(x,bins=min(60,max(5,int(np.sqrt(len(x))))),density=True);ax.stairs(density,edges,label=label,color=color);return density[density>0]
    ax.axvline(float(np.mean(x)),label=f"{label} (constant)",color=color,linestyle="--")
    return np.array([])

def use_log_density_if_needed(ax,density_values):
    positive=np.concatenate([x for x in density_values if len(x)]) if any(len(x) for x in density_values) else np.array([])
    if len(positive) and positive.max()/positive.min()>100:
        ax.set_yscale("log");ax.set_ylim(bottom=max(positive.min()*.5,positive.max()*1e-6));return True
    return False

def robust_limits(values,lower=.5,upper=99.5):
    x=finite(values)
    if not len(x): return None
    lo,hi=np.percentile(x,[lower,upper])
    if hi<=lo: return None
    pad=.04*(hi-lo); return float(lo-pad),float(hi+pad)

def add_full_range_inset(ax,x,y,groups):
    xlim=robust_limits(x);ylim=robust_limits(y)
    if xlim is None or ylim is None:return
    fullx=(np.nanmin(x),np.nanmax(x));fully=(np.nanmin(y),np.nanmax(y))
    clipped=(fullx[0]<xlim[0] or fullx[1]>xlim[1] or fully[0]<ylim[0] or fully[1]>ylim[1])
    ax.set_xlim(xlim);ax.set_ylim(ylim)
    if not clipped:return
    iax=inset_axes(ax,width="30%",height="30%",loc="upper right",borderpad=1)
    for gx,gy,color in groups: iax.scatter(gx,gy,s=2,alpha=.2,color=color)
    iax.set_title("Full range",fontsize=7);iax.tick_params(labelsize=6)

def bond_reports(bond,root,cfg,edge_bond=None):
    props=[c for c in bond if c.startswith(("q","what","mean_s6","fraction_s6"))]
    summaries=[]
    for system,g in bond.groupby("system"):
        for prop in props: summaries.append(dict(system=system,property=prop,**distribution_summary(g[prop])))
    summary=pd.DataFrame(summaries); atomic_csv(root/"combined_results"/"bond_order_summaries.csv",summary)
    # One independent mean per ruby simulation.
    wide=summary.pivot(index="system",columns="property",values="mean").reset_index(); wide["angle"]=wide.system.str.extract(r"ruby_(\d+deg)")
    tests=[]
    for prop in props:
        result=pairwise_replicate_tests(wide[["angle",prop]].rename(columns={prop:"value"}),"value",[],cfg)
        if len(result): result.insert(0,"property",prop);tests.append(result)
    atomic_csv(root/"combined_results"/"bond_order_angle_tests.csv",pd.concat(tests,ignore_index=True) if tests else pd.DataFrame())
    distances=[]
    crystals=[s for s in bond.system.unique() if s.startswith("crystal_")]
    for prop in props:
        for angle in cfg["angles"]:
            ruby=finite(bond.loc[bond.system.str.startswith(f"ruby_{angle}"),prop])
            for crystal in crystals:
                ideal=finite(bond.loc[bond.system==crystal,prop])
                if len(ruby) and len(ideal): distances.append(dict(angle=angle,crystal=crystal.replace("crystal_",""),property=prop,wasserstein=stats.wasserstein_distance(ruby,ideal),jensen_shannon=js_distance(ruby,ideal),standardized_mean_difference=(ruby.mean()-ideal.mean())/ideal.std(ddof=1) if ideal.std(ddof=1)>0 else np.nan))
    atomic_csv(root/"combined_results"/"ruby_crystal_bond_distances.csv",pd.DataFrame(distances))
    plot_dir=root/"combined_results"/"bond_order_plots";plot_dir.mkdir(exist_ok=True)
    palette={**geometry_colors(cfg),"SC":"#2CA02C","BCC":"#9467BD","FCC":"#8C564B","HCP":"#E377C2"}
    systems=[(angle,bond.system.str.startswith(f"ruby_{angle}"),palette[angle]) for angle in cfg["angles"]]+[(c.replace("crystal_",""),bond.system==c,palette[c.replace("crystal_","")]) for c in crystals]
    for prop in ("q4","q6","qbar4","qbar6","mean_s6"):
        if prop not in bond:continue
        fig,ax=plt.subplots(figsize=(7,4.5))
        density_values=[]
        for label,mask,color in systems:
            density_values.append(plot_hist_safe(ax,bond.loc[mask,prop],label,color))
        is_log=use_log_density_if_needed(ax,density_values);ax.set(xlabel=prop,ylabel="Density"+(" (log scale)" if is_log else ""),title=f"{prop} distributions"+(" — log density" if is_log else ""));ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(plot_dir/f"{prop}_distributions.png",dpi=220);plt.close(fig)
    if edge_bond is not None and len(edge_bond):
        fig,ax=plt.subplots(figsize=(7,4.5))
        density_values=[]
        for label in cfg["angles"]:
            prefix=f"ruby_{label}";color=palette[label]
            density_values.append(plot_hist_safe(ax,edge_bond.loc[edge_bond.system.str.startswith(prefix),"s6"],label,color))
        for crystal in sorted(s for s in edge_bond.system.unique() if s.startswith("crystal_")): density_values.append(plot_hist_safe(ax,edge_bond.loc[edge_bond.system==crystal,"s6"],crystal.replace("crystal_",""),palette[crystal.replace("crystal_","")]))
        is_log=use_log_density_if_needed(ax,density_values);ax.set(xlabel="Contact S6",ylabel="Density"+(" (log scale)" if is_log else ""),title="Bond-order correlation"+(" — log density" if is_log else ""));ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(plot_dir/"contact_s6_distributions.png",dpi=220);plt.close(fig)
    for xprop,yprop in (("q4","q6"),("what4","q6"),("what6","q6"),("coordination","q6")):
        fig,ax=plt.subplots(figsize=(6,5))
        plotted=[]
        for label,mask,color in systems[:len(cfg["angles"])]:
            sample=bond.loc[mask,[xprop,yprop]].dropna();sample=sample.sample(min(5000,len(sample)),random_state=20260804);ax.scatter(sample[xprop],sample[yprop],s=5,alpha=.25,label=label,color=color);plotted.append((sample[xprop].to_numpy(),sample[yprop].to_numpy(),color))
        allx=np.concatenate([p[0] for p in plotted]);ally=np.concatenate([p[1] for p in plotted]);add_full_range_inset(ax,allx,ally,plotted)
        ax.set(xlabel=xprop,ylabel=yprop);ax.legend(frameon=False,loc="best");fig.tight_layout();fig.savefig(plot_dir/f"{xprop}_vs_{yprop}.png",dpi=220);plt.close(fig)
    # PCA and normalized-property heatmap of system means.
    features=[p for p in props if p in wide and wide[p].notna().all()]; X=wide[features].to_numpy(float); X=(X-X.mean(0))/np.where(X.std(0)>0,X.std(0),1);u,s,_=np.linalg.svd(X,full_matrices=False)
    geometry_codes={a:i for i,a in enumerate(cfg["angles"])};codes=wide.angle.map(geometry_codes).fillna(len(geometry_codes)).to_numpy();fig,ax=plt.subplots(figsize=(6,5));ax.scatter(u[:,0]*s[0],u[:,1]*s[1],c=codes,cmap="viridis");ax.set(xlabel="PC1",ylabel="PC2",title="Bond-order system PCA");fig.tight_layout();fig.savefig(plot_dir/"bond_order_pca.png",dpi=220);plt.close(fig)
    means=wide.groupby(wide.angle.fillna(wide.system.str.replace("crystal_","",regex=False)))[features].mean();z=(means-means.mean())/means.std().replace(0,1);fig,ax=plt.subplots(figsize=(max(8,len(features)*.6),5));im=ax.imshow(z,aspect="auto",cmap="coolwarm");ax.set_xticks(range(len(features)),features,rotation=60,ha="right");ax.set_yticks(range(len(z)),z.index);fig.colorbar(im,ax=ax,label="Standardized mean");fig.tight_layout();fig.savefig(plot_dir/"bond_order_heatmap.png",dpi=220);plt.close(fig)
    # Representative 3-D particle plots colored by q4, q6, and qbar6.
    representatives=[s for s in [next((s for s in bond.system.unique() if s.startswith(f"ruby_{a}")),None) for a in cfg["angles"]]+crystals if s is not None]
    for prop in ("q4","q6","qbar6"):
        color_values=finite(bond.loc[bond.system.isin(representatives),prop]);vmin,vmax=np.percentile(color_values,[1,99]);norm=Normalize(vmin=vmin,vmax=vmax)
        fig=plt.figure(figsize=(4*len(representatives),4))
        for i,system in enumerate(representatives,1):
            ax=fig.add_subplot(1,len(representatives),i,projection="3d");g=bond[bond.system==system];sc=ax.scatter(g.x,g.y,g.z,c=g[prop],s=4,cmap="viridis",norm=norm);ax.set_title(system)
        fig.subplots_adjust(right=.92,wspace=.1);cax=fig.add_axes([.94,.2,.012,.6]);fig.colorbar(sc,cax=cax,label=f"{prop} (shared 1st–99th percentile scale)");fig.savefig(plot_dir/f"representative_3d_{prop}.png",dpi=180,bbox_inches="tight");plt.close(fig)

def local_crystal_distances(root,j2_raw,j3_raw,cfg):
    if not j2_raw or not j3_raw:return
    ruby=pd.concat([pd.read_csv(p) for p in j2_raw],ignore_index=True); crystal=pd.concat([pd.read_csv(p) for p in j3_raw],ignore_index=True); rows=[]
    idr={"angle","sim_idx","hop","bundle","center_node"};idc={"structure","hop","center_node"}
    for hop in sorted(set(ruby.hop)&set(crystal.hop)):
        r=ruby[ruby.hop==hop];c=crystal[crystal.hop==hop]
        for prop in sorted((set(r)-idr)&(set(c)-idc)):
            for angle in cfg["angles"]:
                a=finite(r.loc[r.angle==angle,prop])
                for structure in sorted(c.structure.unique()):
                    b=finite(c.loc[c.structure==structure,prop])
                    if len(a) and len(b): rows.append(dict(hop=hop,property=prop,angle=angle,structure=structure,wasserstein=stats.wasserstein_distance(a,b),jensen_shannon=js_distance(a,b),standardized_mean_difference=(a.mean()-b.mean())/b.std(ddof=1) if b.std(ddof=1)>0 else np.nan))
    atomic_csv(root/"combined_results"/"ruby_crystal_local_distances.csv",pd.DataFrame(rows))

def _contrast_table(summary,identifiers,statistics):
    rows=[]
    for keys,g in summary.groupby(identifiers,dropna=False):
        key=dict(zip(identifiers,keys if isinstance(keys,tuple) else (keys,)))
        for stat_name in statistics:
            a=finite(g.loc[g.group=="non_high_force",stat_name]); b=finite(g.loc[g.group=="high_force",stat_name])
            if len(a) and len(b): rows.append({**key,"statistic":stat_name,"non_high_force":a[0],"high_force":b[0],"difference_high_minus_non":b[0]-a[0]})
    return pd.DataFrame(rows)

def _replicate_comparisons(contrast,group_columns,cfg):
    rows=[]
    for keys,g in contrast.groupby(group_columns,dropna=False):
        key=dict(zip(group_columns,keys if isinstance(keys,tuple) else (keys,)))
        for angle in cfg["angles"]:
            q=g[g.angle==angle] if "angle" in g else g
            if len(q)>=2: rows.append({**key,"comparison":f"high_vs_non_within_{angle}","angle":angle,**replicate_test(q.non_high_force,q.high_force,paired=True)})
        if "angle" in g:
            for geometry_a,geometry_b in geometry_pairs(cfg):
                a=g.loc[g.angle==geometry_a,"difference_high_minus_non"]; b=g.loc[g.angle==geometry_b,"difference_high_minus_non"]
                if len(finite(a))>=2 and len(finite(b))>=2: rows.append({**key,"comparison":f"high_minus_non_{geometry_b}_vs_{geometry_a}","angle":"between_geometries","geometry_a":geometry_a,"geometry_b":geometry_b,**replicate_test(a,b,paired=False)})
    return pd.DataFrame(rows)

def _heatmap(frame,row_col,column_col,value_col,path,title):
    if frame.empty:return
    table=frame.pivot_table(index=row_col,columns=column_col,values=value_col,aggfunc="mean"); values=table.to_numpy(float); finite_values=values[np.isfinite(values)]
    if not len(finite_values):return
    bound=np.percentile(np.abs(finite_values),95) or 1; fig,ax=plt.subplots(figsize=(max(7,.8*len(table.columns)),max(4,.28*len(table))))
    im=ax.imshow(values,aspect="auto",cmap="coolwarm",vmin=-bound,vmax=bound);ax.set_xticks(range(len(table.columns)),table.columns,rotation=45,ha="right");ax.set_yticks(range(len(table)),table.index,fontsize=7);ax.set_title(title);fig.colorbar(im,ax=ax,label=value_col);fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def job5_complete_distribution_plots(root,plot,cfg):
    colors={"non_high_force":"#2474B5","high_force":"#D95F02"}
    for scope in ("node","edge"):
        paths=sorted((root/"job5_high_force_comparison").glob(f"*_complete_{scope}s.csv"))
        if not paths:continue
        frames=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);props=[c for c in frames if c not in ID_COLUMNS and c!="load_step"];(plot/f"{scope}_distributions").mkdir(exist_ok=True)
        for prop in props:
            fig,axes=plt.subplots(1,len(cfg["angles"]),figsize=(5*len(cfg["angles"]),4),sharex=True,squeeze=False)
            for ax,angle in zip(axes.flat,cfg["angles"]):
                density=[];angle_frame=frames[frames.angle==angle]
                for group in ("non_high_force","high_force"):
                    values=angle_frame.loc[angle_frame.group==group,prop];density.append(plot_hist_safe(ax,values,group.replace("_"," "),colors[group]))
                is_log=use_log_density_if_needed(ax,density);ax.set(title=angle,xlabel=prop,ylabel="Density"+(" (log)" if is_log else ""))
                if ax.get_legend_handles_labels()[0]:ax.legend(frameon=False)
            fig.tight_layout();fig.savefig(plot/f"{scope}_distributions"/f"{prop}.png",dpi=180);plt.close(fig)

def job5_reports(cfg,root,j5_complete,j5_centered):
    out=root/"combined_results"; plot=out/"job5_high_force_plots"; plot.mkdir(exist_ok=True)
    stats_names=["mean","median","std","iqr","mad","p05","p10","p25","p75","p90","p95","p99","skewness","kurtosis","fwhm","entropy"]
    if j5_complete:
        summary=pd.concat([pd.read_csv(p) for p in j5_complete],ignore_index=True);atomic_csv(out/"high_force_complete_group_statistics.csv",summary)
        contrast=_contrast_table(summary,["angle","sim_idx","load_step","scope","property"],stats_names);atomic_csv(out/"high_force_complete_simulation_contrasts.csv",contrast)
        tests=_replicate_comparisons(contrast,["scope","property","statistic"],cfg);atomic_csv(out/"high_force_complete_statistical_comparisons.csv",tests)
        pooled=[]
        for scope in ("node","edge"):
            paths=sorted((root/"job5_high_force_comparison").glob(f"*_complete_{scope}s.csv"))
            if not paths: continue
            frame=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True)
            for prop in [c for c in frame if c not in ID_COLUMNS and c!="load_step"]:
                for (angle,group),g in frame.groupby(["angle","group"]): pooled.append(dict(scope=scope,property=prop,angle=angle,group=group,**distribution_summary(g[prop])))
        atomic_csv(out/"high_force_complete_pooled_statistics.csv",pd.DataFrame(pooled))
        colors={"non_high_force":"#2474B5","high_force":"#D95F02"}
        job5_complete_distribution_plots(root,plot,cfg)
        for scope in ("node","edge"):
            paths=sorted((root/"job5_high_force_comparison").glob(f"*_complete_{scope}s.csv"))
            if not paths:continue
            frames=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True); props=[c for c in frames if c not in ID_COLUMNS and c!="load_step"]
            (plot/f"{scope}_distributions").mkdir(exist_ok=True);(plot/f"{scope}_simulation_points").mkdir(exist_ok=True)
            for prop in props:
                means=summary[(summary.scope==scope)&(summary.property==prop)][["angle","sim_idx","group","mean"]]; cats=[(angle,group) for angle in cfg["angles"] for group in ("non_high_force","high_force")]
                fig,ax=plt.subplots(figsize=(7,4));rng=np.random.default_rng(cfg["random_seed"])
                for i,(angle,group) in enumerate(cats):
                    vals=finite(means.loc[(means.angle==angle)&(means.group==group),"mean"]); parts=ax.violinplot(vals,[i],showextrema=False) if len(vals)>1 else None
                    if parts:
                        for body in parts["bodies"]:body.set_facecolor(colors[group]);body.set_alpha(.25)
                    ax.scatter(i+rng.uniform(-.08,.08,len(vals)),vals,s=15,color=colors[group],alpha=.8)
                ax.set_xticks(range(len(cats)),[f"{a} {'non' if g=='non_high_force' else 'high'}" for a,g in cats],rotation=45,ha="right");ax.set(ylabel=f"simulation mean {prop}",title=f"{scope}: {prop}");fig.tight_layout();fig.savefig(plot/f"{scope}_simulation_points"/f"{prop}.png",dpi=180);plt.close(fig)
        if not tests.empty:
            effect=tests[(tests.statistic=="mean")&tests.comparison.str.startswith("high_vs")];_heatmap(effect,"property","angle","effect_size",plot/"complete_property_effect_heatmap.png","High-force effect sizes within each angle")
            angle_effect=tests[(tests.statistic=="mean")&tests.comparison.str.startswith("high_minus_non_")];_heatmap(angle_effect,"property","comparison","effect_size",plot/"complete_geometry_difference_heatmap.png","Geometry effect on high-force minus non-high-force difference")
    if j5_centered:
        summary=pd.concat([pd.read_csv(p) for p in j5_centered],ignore_index=True);atomic_csv(out/"high_force_centered_group_statistics.csv",summary)
        contrast=_contrast_table(summary,["angle","sim_idx","load_step","hop","property"],stats_names);atomic_csv(out/"high_force_centered_simulation_contrasts.csv",contrast)
        tests=_replicate_comparisons(contrast,["hop","property","statistic"],cfg);atomic_csv(out/"high_force_centered_statistical_comparisons.csv",tests)
        mean_effect=tests[(tests.statistic=="mean")&tests.comparison.str.startswith("high_vs")].copy() if not tests.empty else pd.DataFrame(columns=["property","angle","hop","effect_size"])
        for prefix,label in (("center__","center_node"),("graph__","subgraph_graph"),("node_","subgraph_averaged_node"),("edge_","subgraph_averaged_edge")):
            subset=mean_effect[mean_effect.property.str.startswith(prefix)];_heatmap(subset,"property",["angle","hop"],"effect_size",plot/f"centered_{label}_effect_heatmap.png",f"{label.replace('_',' ').title()} effect size")
            if subset.empty: continue
            ranking=subset.groupby("property").effect_size.apply(lambda x:np.nanmax(np.abs(x)) if np.isfinite(x).any() else np.nan).nlargest(8).index
            for prop in ranking:
                g=contrast[(contrast.property==prop)&(contrast.statistic=="mean")];fig,ax=plt.subplots(figsize=(6,4))
                for angle,color in geometry_colors(cfg).items():
                    q=g[g.angle==angle];means=q.groupby("hop").difference_high_minus_non.mean();errs=q.groupby("hop").difference_high_minus_non.sem()*1.96;ax.errorbar(means.index,means,yerr=errs,marker="o",label=angle,color=color)
                ax.axhline(0,color="black",lw=.8);ax.set(xlabel="Hop size",ylabel="High − non-high",title=prop);ax.legend(frameon=False);fig.tight_layout();safe=re.sub(r"[^A-Za-z0-9_.-]+","_",prop);fig.savefig(plot/f"centered_{label}_{safe}.png",dpi=180);plt.close(fig)

def job6_reports(cfg,root,cluster_files,summary_files):
    if not cluster_files:return
    out=root/"combined_results";plot=out/"job6_force_cluster_plots";plot.mkdir(exist_ok=True);clusters=pd.concat([pd.read_csv(p) for p in cluster_files],ignore_index=True);summaries=pd.concat([pd.read_csv(p) for p in summary_files],ignore_index=True);atomic_csv(out/"force_clusters.csv",clusters);atomic_csv(out/"force_cluster_simulation_summaries.csv",summaries)
    numeric=[c for c in summaries if c not in {"angle","sim_idx","load_step","cluster_size_distribution","cluster_diameter_distribution","isolated_particles_included"}];tests=[]
    for prop in numeric:
        result=pairwise_replicate_tests(summaries[["angle",prop]].rename(columns={prop:"value"}),"value",[],cfg)
        if len(result):result.insert(0,"property",prop);tests.append(result)
    atomic_csv(out/"force_cluster_angle_tests.csv",pd.concat(tests,ignore_index=True) if tests else pd.DataFrame())
    colors=geometry_colors(cfg)
    for prop,title in (("node_count","Force-cluster size distributions"),("diameter","Cluster diameter distributions"),("aspect_ratio","Cluster aspect ratios"),("principal_axis_loading_alignment","Principal-axis/loading alignment")):
        fig,ax=plt.subplots(figsize=(6,4));density=[]
        for angle in cfg["angles"]:density.append(plot_hist_safe(ax,clusters.loc[clusters.angle==angle,prop],angle,colors[angle]))
        use_log_density_if_needed(ax,density);ax.set(xlabel=prop,ylabel="Pooled descriptive density",title=title);ax.legend(frameon=False);fig.tight_layout();fig.savefig(plot/f"{prop}_distribution.png",dpi=200);plt.close(fig)
    for x,y,name in (("node_count","diameter","size_vs_diameter"),("node_count","total_normal_force","size_vs_total_force")):
        fig,ax=plt.subplots(figsize=(6,4))
        for angle in cfg["angles"]:g=clusters[clusters.angle==angle];ax.scatter(g[x],g[y],s=12,alpha=.35,label=angle,color=colors[angle])
        ax.set(xlabel=x,ylabel=y,title=name.replace("_"," ").title());ax.legend(frameon=False);fig.tight_layout();fig.savefig(plot/f"{name}.png",dpi=200);plt.close(fig)
    for prop in ("cluster_count","mean_cluster_size","largest_cluster_fraction"):
        fig,ax=plt.subplots(figsize=(6,4));data=[finite(summaries.loc[summaries.angle==a,prop]) for a in cfg["angles"]];ax.boxplot(data,labels=cfg["angles"])
        for i,(angle,vals) in enumerate(zip(cfg["angles"],data),1):ax.scatter(np.full(len(vals),i),vals,color=colors[angle],s=18,alpha=.75)
        ax.set(ylabel=prop,title=f"{prop.replace('_',' ').title()} per simulation");fig.tight_layout();fig.savefig(plot/f"simulation_{prop}.png",dpi=200);plt.close(fig)
    topology=["mean_degree","density","diameter_per_sqrt_node","clustering_coefficient","loops_per_node","spectral_radius","algebraic_connectivity","largest_core_fraction"]
    normalized=clusters.groupby("angle")[topology].mean();z=(normalized-normalized.mean())/normalized.std().replace(0,1);fig,ax=plt.subplots(figsize=(9,3));im=ax.imshow(z,aspect="auto",cmap="coolwarm");ax.set_xticks(range(len(topology)),topology,rotation=45,ha="right");ax.set_yticks(range(len(z)),z.index);fig.colorbar(im,ax=ax,label="standardized pooled mean");fig.tight_layout();fig.savefig(plot/"normalized_topology_comparison.png",dpi=220);plt.close(fig)
    from job6_force_cluster_analysis import unwrap_component,geometry as force_geometry
    graphs=load_graph_dict(cfg);box_lengths,periodic_axes=force_geometry(cfg,root);ncols=min(2,len(cfg["angles"]));nrows=int(np.ceil(len(cfg["angles"])/ncols));fig=plt.figure(figsize=(5*ncols,4*nrows))
    for panel,angle in enumerate(cfg["angles"],1):
        ax=fig.add_subplot(nrows,ncols,panel,projection="3d");available=clusters[clusters.angle==angle].sort_values("node_count",ascending=False)
        if available.empty: ax.set_title(f"{angle}: no completed pilot");continue
        largest=available.iloc[0];G=graphs[angle][cfg["primary_graph_view"]][int(largest.sim_idx)];nodes=json.loads(largest.node_ids);nodes=[int(n) if str(n).isdigit() else n for n in nodes];node_set=set(nodes);high_edges=[(u,v) for u,v,d in G.edges(data=True) if u in node_set and v in node_set and truthy_label(d.get(cfg["high_force_edge_label"],False))];H=G.edge_subgraph(high_edges).copy();coords=unwrap_component(H,box_lengths,periodic_axes);xyz=np.asarray([coords[n] for n in H]);ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=12,color=colors[angle]);ax.set_title(f"{angle}: largest cluster, n={len(nodes)} (unwrapped)")
    fig.tight_layout();fig.savefig(plot/"representative_3d_force_clusters.png",dpi=200);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--allow-incomplete",action="store_true"); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config); root=ensure_layout(cfg); log=setup_logging(cfg,"merge")
    j2=current_job2_files(root/"job2_subgraphs",summary=True,cfg=cfg); j3=sorted((root/"job3_crystal_baselines").glob("*_summary.csv")); j4n=sorted(p for p in (root/"job4_bond_order").glob("*_nodes.csv") if not p.name.startswith("validation_")); j4e=sorted((root/"job4_bond_order").glob("*_edges.csv")); j4s=sorted((root/"job4_bond_order").glob("*_system.json"))
    j5complete=sorted((root/"job5_high_force_comparison").glob("*_complete_summary.csv"));j5centered=sorted((root/"job5_high_force_comparison").glob("*_centered_hop*_summary.csv"));j6clusters=sorted((root/"job6_force_clusters").glob("*_clusters.csv"));j6summary=sorted((root/"job6_force_clusters").glob("*_simulation_summary.csv"))
    nsim=len(cfg["angles"])*cfg["simulations_per_angle"];expected={"job2":nsim*len(cfg["hop_sizes"])*len(cfg["job2_property_bundles"]),"job3":4*len(cfg["hop_sizes"]),"job4_nodes":nsim+4,"job5_complete":nsim,"job5_centered":nsim*len(cfg["hop_sizes"]),"job6":nsim}
    found={"job2":len(j2),"job3":len(j3),"job4_nodes":len(j4n),"job5_complete":len(j5complete),"job5_centered":len(j5centered),"job6":len(j6clusters)}; atomic_json(root/"combined_results"/"completion_audit.json",{"expected":expected,"found":found})
    if args.dry_run: print({"expected":expected,"found":found}); return
    if not args.allow_incomplete and any(found[k]<expected[k] for k in expected): raise SystemExit(f"Incomplete tasks: expected={expected}, found={found}")
    if j2:
        local=pd.concat([pd.read_csv(p) for p in j2],ignore_index=True); atomic_csv(root/"combined_results"/"local_subgraph_summaries.csv",local)
        tests=pairwise_replicate_tests(local,"mean",["hop","property"],cfg)
        atomic_csv(root/"combined_results"/"local_angle_tests.csv",tests)
        complete_path=root/"job1_global_figures"/"simulation_topology.csv"
        if complete_path.exists():
            complete=pd.read_csv(complete_path); comparisons=[]
            mapping={"mean_degree":"mean_degree","mean_closeness":"mean_closeness","mean_clustering":"mean_clustering"}
            for local_prop,global_prop in mapping.items():
                subset=local[(local.property==local_prop)&(local.bundle.isin(["fast","paths"]))]
                for row in subset.itertuples():
                    match=complete[(complete.angle==row.angle)&(complete.sim_idx==row.sim_idx)]
                    if len(match): comparisons.append(dict(angle=row.angle,sim_idx=row.sim_idx,hop=row.hop,property=local_prop,local_mean=row.mean,complete_graph_value=float(match.iloc[0][global_prop]),local_minus_complete=row.mean-float(match.iloc[0][global_prop])))
            atomic_csv(root/"combined_results"/"local_vs_complete_graph.csv",pd.DataFrame(comparisons))
    if j3: atomic_csv(root/"combined_results"/"crystal_local_summaries.csv",pd.concat([pd.read_csv(p) for p in j3],ignore_index=True))
    if j4n:
        bond=pd.concat([pd.read_csv(p) for p in j4n],ignore_index=True); atomic_csv(root/"combined_results"/"bond_order_nodes.csv",bond)
        edge_bond=pd.concat([pd.read_csv(p) for p in j4e],ignore_index=True) if j4e else None
        if edge_bond is not None: atomic_csv(root/"combined_results"/"bond_order_edges.csv",edge_bond)
        bond_reports(bond,root,cfg,edge_bond)
    if j4s:
        rows=[]
        for p in j4s: rows.append({"system":p.name.replace("_system.json",""),**json.loads(p.read_text())})
        atomic_csv(root/"combined_results"/"bond_order_system_values.csv",pd.DataFrame(rows))
    j2raw=current_job2_files(root/"job2_subgraphs",summary=False,cfg=cfg);j3raw=sorted(p for p in (root/"job3_crystal_baselines").glob("*.csv") if not p.name.endswith("_summary.csv"));local_crystal_distances(root,j2raw,j3raw,cfg)
    job5_reports(cfg,root,j5complete,j5centered);job6_reports(cfg,root,j6clusters,j6summary)
    from plot_basic_high_force_systems import generate_views
    generate_views(cfg,root)
    log.info("Merge complete; found=%s",found)
if __name__=="__main__":main()
