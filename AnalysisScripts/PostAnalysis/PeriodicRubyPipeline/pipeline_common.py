#!/usr/bin/env python3
"""Shared, dependency-light helpers for the periodic ruby analysis pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import tempfile
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def load_config(path=CONFIG_PATH):
    # JSON is a strict subset of YAML, avoiding a PyYAML dependency on compute nodes.
    raw = Path(path).read_text()
    cfg = json.loads(raw)
    root = Path(cfg["project_root"])
    for key in ("graph_pickle", "existing_results", "output_root"):
        value = Path(cfg[key])
        cfg[key] = value if value.is_absolute() else root / value
    cfg["config_path"] = Path(path).resolve()
    cfg["config_hash"] = hashlib.sha256(raw.encode()).hexdigest()
    code_digest=hashlib.sha256()
    for source in sorted(HERE.glob("*.py")):
        code_digest.update(source.name.encode()); code_digest.update(source.read_bytes())
    cfg["code_hash"] = code_digest.hexdigest()
    return cfg


def ensure_layout(cfg):
    root = Path(cfg["output_root"])
    for name in (
        "job0_metadata", "job1_global_figures", "job2_subgraphs",
        "job3_crystal_baselines", "job4_bond_order", "job5_high_force_comparison",
        "job6_force_clusters", "combined_results", "logs",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    snapshot = root / "config.snapshot.json"
    atomic_json(snapshot, serializable_config(cfg))
    return root


def serializable_config(cfg):
    return {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}


def setup_logging(cfg, job_name, task_id=None):
    ensure_layout(cfg)
    suffix = "" if task_id is None else f"_{task_id}"
    path = Path(cfg["output_root"]) / "logs" / f"{job_name}{suffix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(job_name)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n"); tmp = handle.name
    os.replace(tmp, path)


def atomic_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), suffix=".csv", delete=False) as handle:
        tmp = handle.name
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def completion_path(output):
    return Path(str(output) + ".complete.json")


def output_valid(output, cfg, required_columns=()):
    output = Path(output); manifest = completion_path(output)
    if not output.exists() or output.stat().st_size == 0 or not manifest.exists(): return False
    try:
        info = json.loads(manifest.read_text())
        if info.get("config_hash") != cfg["config_hash"] or info.get("code_hash") != cfg["code_hash"]: return False
        if required_columns and output.suffix == ".csv":
            columns = set(pd.read_csv(output, nrows=0).columns)
            if not set(required_columns).issubset(columns): return False
        return True
    except Exception:
        return False


def mark_complete(output, cfg, extra=None):
    info = {"config_hash": cfg["config_hash"], "code_hash": cfg["code_hash"], "output": str(output), "size": Path(output).stat().st_size}
    if extra: info.update(extra)
    atomic_json(completion_path(output), info)


def load_graph_dict(cfg):
    with open(cfg["graph_pickle"], "rb") as handle:
        return pickle.load(handle)


def graph_index(graph_dict, cfg):
    return [(a, i) for a in cfg["angles"] for i in range(len(graph_dict[a][cfg["primary_graph_view"]]))]


def geometry_pairs(cfg):
    """All unordered geometry pairs, in configured display order."""
    return list(combinations(cfg["angles"], 2))


def geometry_colors(cfg):
    """Stable, distinct colors for any number of configured geometries."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if len(cfg["angles"]) <= 10 else "tab20")
    return {angle: cmap(i % cmap.N) for i, angle in enumerate(cfg["angles"])}


def pairwise_replicate_tests(frame, value, group_columns, cfg, geometry_column="angle"):
    """Compare every geometry pair using simulation-level values only."""
    rows = []
    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip(group_columns, keys))
        for geometry_a, geometry_b in geometry_pairs(cfg):
            a = group.loc[group[geometry_column] == geometry_a, value]
            b = group.loc[group[geometry_column] == geometry_b, value]
            if len(finite(a)) >= 2 and len(finite(b)) >= 2:
                rows.append({
                    **identifiers,
                    "geometry_a": geometry_a,
                    "geometry_b": geometry_b,
                    "comparison": f"{geometry_b}_minus_{geometry_a}",
                    **replicate_test(a, b, paired=False),
                })
    return pd.DataFrame(rows)


def task_id(default=None):
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(value) if value is not None else default


def unsuffixed(keys, cfg):
    suffixes = tuple(cfg.get("exclude_property_suffixes", []))
    return sorted(k for k in keys if not k.endswith(suffixes))


def finite(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return x[np.isfinite(x)]


def scalar_numeric_properties(records, cfg, excluded=()):
    """Return scalar numeric/bool attributes present in at least one record."""
    records=list(records); blocked=set(excluded); suffixes=tuple(cfg.get("exclude_property_suffixes", [])); found=[]
    for key in sorted(set().union(*(d.keys() for d in records)) if records else set()):
        if key in blocked or key.endswith(suffixes): continue
        values=[d.get(key) for d in records if d.get(key) is not None]
        if any(isinstance(v,(bool,np.bool_,int,float,np.integer,np.floating)) and np.ndim(v)==0 for v in values): found.append(key)
    return found


def truthy_label(value):
    if isinstance(value,str): return value.strip().lower() in {"1","true","yes","high","high_force"}
    try: return bool(value)
    except Exception: return False


def high_force_labels(G,cfg):
    """Reuse stored labels; generate node labels from high-force edges only if absent."""
    edge_key=cfg["high_force_edge_label"]; node_key=cfg["high_force_node_label"]
    edge_labels={(u,v):truthy_label(d.get(edge_key,False)) for u,v,d in G.edges(data=True)}
    stored=all(node_key in d for _,d in G.nodes(data=True))
    if stored: node_labels={n:truthy_label(d[node_key]) for n,d in G.nodes(data=True)}
    else:
        high_nodes={n for edge,is_high in edge_labels.items() if is_high for n in edge}
        node_labels={n:n in high_nodes for n in G.nodes()}
    return node_labels,edge_labels,("stored" if stored else cfg["high_force_node_fallback_rule"])


def replicate_test(a,b,paired=False):
    """Tests/effect size for replicate-level arrays, oriented b minus a."""
    a=finite(a); b=finite(b); out={"n_a":len(a),"n_b":len(b),"mean_a":np.mean(a) if len(a) else np.nan,"mean_b":np.mean(b) if len(b) else np.nan}
    if paired:
        n=min(len(a),len(b)); a=a[:n]; b=b[:n]; out["n_pairs"]=n
        if n>=2:
            t=stats.ttest_rel(b,a,nan_policy="omit"); out.update(test="paired_t",test_statistic=t.statistic,test_p=t.pvalue)
            try: w=stats.wilcoxon(b,a); out.update(nonparametric_test="wilcoxon",nonparametric_statistic=w.statistic,nonparametric_p=w.pvalue)
            except ValueError: out.update(nonparametric_test="wilcoxon",nonparametric_statistic=np.nan,nonparametric_p=np.nan)
            delta=b-a; sd=delta.std(ddof=1); out["effect_size"]=delta.mean()/sd if sd>0 else 0.0
    elif len(a)>=2 and len(b)>=2:
        t=stats.ttest_ind(b,a,equal_var=False); u=stats.mannwhitneyu(b,a,alternative="two-sided")
        df=len(a)+len(b)-2; pooled=((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/df
        out.update(test="welch_t",test_statistic=t.statistic,test_p=t.pvalue,nonparametric_test="mann_whitney_u",nonparametric_statistic=u.statistic,nonparametric_p=u.pvalue,effect_size=(b.mean()-a.mean())/np.sqrt(pooled)*(1-3/(4*df-1)) if pooled>0 else 0.0)
    return out


def distribution_summary(values):
    x = finite(values)
    names = ["n", "mean", "median", "min", "max", "std", "variance", "iqr", "mad",
             "p10_p90_width", "fwhm", "cv", "p05", "p10", "p25", "p75", "p90", "p95",
             "p99", "top10_mean", "skewness", "kurtosis", "mode", "peak_density", "entropy",
             "bimodality_coefficient", "n_peaks", "multimodal"]
    out = {k: np.nan for k in names}; out["n"] = int(x.size); out["multimodal"] = False
    if not x.size: return out
    q = np.percentile(x, [5, 10, 25, 75, 90, 95, 99]); mean=float(x.mean())
    std=float(x.std(ddof=1)) if x.size>1 else np.nan; med=float(np.median(x))
    variable=np.ptp(x)>100*np.finfo(float).eps*max(1.0,float(np.max(np.abs(x))))
    skew=float(stats.skew(x,bias=False)) if x.size>=3 and variable else np.nan
    kurt=float(stats.kurtosis(x,fisher=True,bias=False)) if x.size>=4 and variable else np.nan
    out.update(mean=mean,median=med,min=float(x.min()),max=float(x.max()),std=std,
               variance=float(x.var(ddof=1)) if x.size>1 else np.nan,iqr=float(q[3]-q[2]),
               mad=float(np.median(np.abs(x-med))),p10_p90_width=float(q[4]-q[1]),
               cv=std/abs(mean) if np.isfinite(std) and mean else np.nan,p05=q[0],p10=q[1],p25=q[2],
               p75=q[3],p90=q[4],p95=q[5],p99=q[6],top10_mean=float(x[x>=q[4]].mean()),
               skewness=skew,kurtosis=kurt,
               bimodality_coefficient=(skew*skew+1)/(kurt+3) if np.isfinite(skew) and np.isfinite(kurt) and kurt+3>0 else np.nan)
    if x.size>1 and variable:
        # A bounded square-root rule is stable for nearly constant ideal-lattice
        # values where floating-point noise can break automatic FD bin widths.
        counts,_=np.histogram(x,bins=min(128,max(1,int(np.sqrt(x.size))))); p=counts[counts>0]/counts.sum(); out["entropy"]=float(stats.entropy(p))
    elif x.size:
        out["entropy"]=0.0
    if x.size>=5 and np.unique(x).size>=3 and variable:
        try:
            kde=stats.gaussian_kde(x); grid=np.linspace(x.min(),x.max(),512); den=kde(grid)
            peaks,_=find_peaks(den,prominence=max(den.max()*.05,np.finfo(float).eps),distance=10)
            if not peaks.size: peaks=np.array([int(np.argmax(den))])
            dominant=int(peaks[np.argmax(den[peaks])]); out.update(mode=float(grid[dominant]),peak_density=float(den[dominant]),n_peaks=int(peaks.size),multimodal=bool(peaks.size>1))
            if peaks.size==1:
                half=den[dominant]/2; left=np.where(den[:dominant]<=half)[0]; right=np.where(den[dominant+1:]<=half)[0]
                if left.size and right.size: out["fwhm"]=float(grid[dominant+1+right[0]]-grid[left[-1]])
        except (ValueError,np.linalg.LinAlgError): pass
    return out


def particle_positions(G):
    nodes=[]; positions=[]
    for n,d in G.nodes(data=True):
        if not d.get("is_wall",False) and "position" in d:
            nodes.append(n); positions.append(d["position"])
    return nodes,np.asarray(positions,float)


def minimum_image(vectors, box_lengths, periodic_axes):
    result=np.asarray(vectors,float).copy()
    for axis in periodic_axes:
        length=float(box_lengths[axis])
        result[...,axis]-=length*np.round(result[...,axis]/length)
    return result


def core_graph(G):
    return G.subgraph([n for n,d in G.nodes(data=True) if not d.get("is_wall",False)]).copy()


def safe_graph_metrics(H, cfg, bundle="all"):
    n=H.number_of_nodes(); e=H.number_of_edges(); result={"node_count":n,"edge_count":e}
    if not n: return result
    result.update(mean_degree=2*e/n,density=nx.density(H),mean_clustering=float(np.mean(list(nx.clustering(H).values()))) if n else np.nan,
                  triangles=int(sum(nx.triangles(H).values())//3))
    if bundle in ("all","paths"):
        result["mean_closeness"]=float(np.mean(list(nx.closeness_centrality(H).values()))) if n>1 else np.nan
        k=min(cfg.get("job2_betweenness_approx_k",32),n)
        result["mean_betweenness"]=float(np.mean(list(nx.betweenness_centrality(H,k=k if k<n else None,seed=cfg["random_seed"]).values()))) if n>2 else np.nan
        try: result["assortativity"]=float(nx.degree_assortativity_coefficient(H)) if e else np.nan
        except Exception: result["assortativity"]=np.nan
    if bundle in ("all","loops"):
        # Fundamental cycle basis is O(V+E); exact minimum-cycle bases for every
        # overlapping 5-hop neighborhood are prohibitively repetitive.
        cycles=nx.cycle_basis(H) if n>=3 and e else []
        result.update(loop_total=len(cycles),loop_3=sum(len(c)==3 for c in cycles),loop_4=sum(len(c)==4 for c in cycles))
        for size in range(3,11): result[f"loop_fraction_{size}"]=sum(len(c)==size for c in cycles)/len(cycles) if cycles else 0.0
    if bundle in ("all","spectral"):
        if n>1:
            adjacency=nx.to_scipy_sparse_array(H,dtype=float,format="csr")
            if n<=12:
                vals=np.linalg.eigvalsh(adjacency.toarray()); lap=np.linalg.eigvalsh(nx.laplacian_matrix(H).toarray()); result["spectral_radius"]=float(np.max(np.abs(vals))); result["algebraic_connectivity"]=float(sorted(lap)[1])
            else:
                result["spectral_radius"]=float(abs(eigsh(adjacency,k=1,which="LM",return_eigenvectors=False)[0]))
                lap=nx.laplacian_matrix(H).astype(float); small=np.sort(eigsh(lap,k=2,which="SM",return_eigenvectors=False)); result["algebraic_connectivity"]=float(small[1])
    if bundle in ("all","connectivity"):
        result["connectivity_exact_computed"]=bool(n<=cfg.get("job2_exact_connectivity_max_nodes",250) and n>1)
        if result["connectivity_exact_computed"]:
            try: result["edge_connectivity"]=nx.edge_connectivity(H)
            except Exception: result["edge_connectivity"]=np.nan
        else: result["edge_connectivity"]=np.nan
    return result
