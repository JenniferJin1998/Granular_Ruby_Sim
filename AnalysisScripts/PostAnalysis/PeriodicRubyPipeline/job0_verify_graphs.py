#!/usr/bin/env python3
"""Job 0: read-only graph inventory and periodic-geometry reconstruction."""
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline_common import *


def estimate_geometry(graph_dict, cfg):
    """Estimate orthogonal box extents and periodic axes from contacting particles."""
    rows=[]
    diameter=float(cfg["particle_diameter"])
    for angle,sim in graph_index(graph_dict,cfg):
        G=graph_dict[angle]["core"][sim]
        nodes,pos=particle_positions(G); index={n:i for i,n in enumerate(nodes)}
        raw=np.asarray([pos[index[v]]-pos[index[u]] for u,v in G.edges() if u in index and v in index])
        spans=np.ptp(pos,axis=0)
        local_norm=np.linalg.norm(raw,axis=1); local=local_norm<3*diameter
        contact_distance=float(np.median(local_norm[local])) if local.any() else diameter
        for axis in range(3):
            seam=np.abs(raw[:,axis])>max(5*diameter,.5*spans[axis])
            # For an orthogonal periodic box, L is raw separation plus/minus the
            # small minimum-image contact component. The coordinate span plus a
            # robust contact diameter is an independent estimate.
            seam_L=np.median(np.abs(raw[seam,axis]))+contact_distance if seam.any() else np.nan
            range_L=spans[axis]+contact_distance
            rows.append(dict(angle=angle,sim_idx=sim,axis=axis,span=spans[axis],contact_distance=contact_distance,
                             seam_contacts=int(seam.sum()),seam_box_estimate=seam_L,range_box_estimate=range_L))
    detail=pd.DataFrame(rows)
    summary=[]
    for axis,g in detail.groupby("axis"):
        has_seams=(g.seam_contacts>0).mean()>=.8
        vals=g.loc[g.seam_contacts>0,"seam_box_estimate"] if has_seams else g.range_box_estimate
        summary.append(dict(axis=int(axis),axis_name="xyz"[axis],periodic=bool(has_seams),box_length=float(vals.median()),
                            box_length_mean=float(vals.mean()),box_length_std=float(vals.std(ddof=1)),n_simulations=len(g),
                            method="seam_contacts" if has_seams else "coordinate_extent_plus_contact_distance"))
    return detail,pd.DataFrame(summary)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=CONFIG_PATH); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
    cfg=load_config(args.config); root=ensure_layout(cfg); log=setup_logging(cfg,"job0")
    if args.dry_run:
        print(f"Would inspect {cfg['graph_pickle']} and write {root/'job0_metadata'}"); return
    graphs=load_graph_dict(cfg); metadata=[]; properties=[]; issues=[]
    for angle,sim in graph_index(graphs,cfg):
        for view in ("core","full"):
            G=graphs[angle][view][sim]; node_keys=set(); edge_keys=set()
            for _,d in G.nodes(data=True): node_keys.update(d)
            for _,_,d in G.edges(data=True): edge_keys.update(d)
            wall_nodes=sum(bool(d.get("is_wall",False)) for _,d in G.nodes(data=True))
            wall_edges=sum(bool(d.get("is_wall_contact",False)) for *_,d in G.edges(data=True))
            metadata.append(dict(graph_file=cfg["graph_pickle"],angle=angle,simulation_id=sim,load_step=cfg["load_step"],view=view,
                                 node_count=G.number_of_nodes(),edge_count=G.number_of_edges(),particle_nodes=G.number_of_nodes()-wall_nodes,
                                 wall_nodes=wall_nodes,particle_wall_edges=wall_edges,positions_available=all("position" in d for _,d in G.nodes(data=True) if not d.get("is_wall",False)),
                                 box_dimensions_stored=any(k in G.graph for k in ("box","box_lengths","dimensions"))))
            for scope,keys in (("node",node_keys),("edge",edge_keys),("graph",set(G.graph))):
                for key in sorted(keys): properties.append(dict(angle=angle,simulation_id=sim,view=view,scope=scope,property=key,primary_eligible=not key.endswith("_with_walls")))
            if view=="core" and wall_nodes: issues.append(dict(angle=angle,simulation_id=sim,issue="core_contains_wall_nodes"))
    detail,geometry=estimate_geometry(graphs,cfg)
    out=root/"job0_metadata"; atomic_csv(out/"graph_metadata.csv",pd.DataFrame(metadata)); atomic_csv(out/"property_inventory.csv",pd.DataFrame(properties).drop_duplicates()); atomic_csv(out/"validation_issues.csv",pd.DataFrame(issues,columns=["angle","simulation_id","issue"])); atomic_csv(out/"geometry_estimates_by_simulation.csv",detail); atomic_csv(out/"geometry_estimate.csv",geometry)
    geo={"box_lengths":{str(r.axis):r.box_length for r in geometry.itertuples()},"periodic_axes":[int(r.axis) for r in geometry.itertuples() if r.periodic],"source":"estimated_from_particle_positions_and_contacts","orthogonal_assumption":True}
    atomic_json(out/"geometry_estimate.json",geo)
    import scipy,networkx,matplotlib
    atomic_json(out/"environment.json",{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"networkx":networkx.__version__,"matplotlib":matplotlib.__version__})
    mark_complete(out/"graph_metadata.csv",cfg,{"graphs":len(metadata)}); log.info("Verified %d graph views; geometry=%s",len(metadata),geo)

if __name__=="__main__": main()
