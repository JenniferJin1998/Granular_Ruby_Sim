# Manuscript visualization

Keep finalized, reproducible manuscript plotting code here. Exploratory plots
and analysis notebooks belong in `PostAnalysis/`. Generated figures belong in
each simulation case's `AnalysisResults/<case>/Visualization/` folder.

## Final-load high-force contact networks

Run from any directory with the existing `graph_analysis` environment:

```bash
/home/yfjin/Research/anaconda3/envs/graph_analysis/bin/python \
  AnalysisScripts/ManuscriptVisualization/plot_final_load_force_contacts.py
```

The default run selects five systems without replacement at each available
orientation, `0deg`, `15deg`, `30deg`, and `45deg`. Seed `20261014` and separate
random streams per orientation make the selection reproducible. System IDs and
pickle-list indices are zero-based; the actual graph `sim_idx` appears in filenames.
Each system gets eight individual figures: with/without walls, each viewed along
x, along y, in 3D, and in 3D with a cylindrical envelope. All use orthographic projection, equal physical axis
scaling, inferno, and an identical 10–80 degree color scale. Angles outside this
range use the endpoint colors; extensions at both colorbar ends indicate clipping.
All contacts remain visible. High-force edges use
a 1.4 pt line width. The two wall variants
share spatial limits. There are no specimen-angle titles or panel letters.

High-force selection uses the input's existing `is_high_force` edge flags. The
default `force_split2` threshold is twice the pooled 0-degree particle-particle
mean normal force, applied to both particle-particle and wall contacts. Use
`--force-split force_split1` for the legacy threshold that includes wall forces
in the reference mean. The cutoff is never recalculated after removing walls.
The wall-free version removes only wall nodes and their incident edges, retaining
all particles, including particles that touch walls.

Edge colors use the stored acute contact-normal angle `angle_with_zz`, in degrees,
relative to the global z axis. No specimen rotation is applied. Particle edges
connect stored particle centers. Wall nodes, represented by triangles, are
located at recorded `contact_location` coordinates, one node per wall contact.
High-force incident nodes use the angle of their strongest visible incident
high-force contact. All other nodes and edges are gray. Original graph data are
read without modification. This script targets the nonperiodic final-load data;
it does not unwrap periodic contacts from other simulation cases.

Output layout (only final individual figures and reproducibility metadata):

```text
AnalysisResults/final_load/Visualization/high_force_contacts/
  force_split2/
    selection_and_render_manifest.json
    0deg/
      sim_003/
        with_walls/
        without_walls/
    15deg/...
    30deg/...
    45deg/...
```

Example filename:

```text
final_load__force_split2__0deg__sim_003__high_force_angle_zz__with_walls__along_x.pdf
```

Each figure is exported as vector PDF, editable-text SVG, and 600 dpi PNG
(160 figures, 480 files). The 4.7-inch working canvas uses 10.5 pt
Helvetica-first labels and 4.5 pt legend markers, preserving the reference
system's original particle sizes, with 1.4 pt high-force edges. Colorbar height
and legend placement follow the projected system bounds for each view; exports
crop unused space above and below the system with a 0.12-inch outer margin.
On systems without Helvetica, the plotting
code falls back to the installed Helvetica-compatible Nimbus Sans font.
The JSON manifest records source SHA-256, threshold provenance when available,
sampled IDs, camera settings, visible node/contact counts and output paths.
Different force splits get separate output folders. Rerunning the same selection
overwrites matching filenames; earlier selections with different names remain.
Use a distinct `--output-root` for a different selection or plotting revision.
Preview or diagnostic outputs should go outside the final results tree.

Useful options: `--seed`, `--samples-per-angle`, `--angles 0deg 45deg`,
`--formats pdf png svg`, `--dpi`, `--input`, `--views`, and `--output-root`.
Figures remain excluded from Git under the repository's existing results policy.
Requires Python 3.10+, NumPy, NetworkX, and Matplotlib 3.6+.

## Cylinder boundary version

The additional `perspective_3d_cylinder` view uses the same camera, force colors,
and node styling as the plain 3D view. A faint transparent side shell, two outer
rims and two silhouette lines indicate a plain open cylinder. There are no end-cap
meshes, smaller circles or radial connecting lines. Its center and radius are
fitted to sidewall contact coordinates (`abs(n_unit[2]) < 0.1`), without inflating
the radius. End rings span the full system height, forming a simple visual
boundary rather than tracing the shaped loading surfaces. Cylinder views use
zoom 1.45, shared with plain views, to make the system larger in the figure.
Geometry is computed from the full graph for both wall variants and recorded
in the render manifest, in millimeters after the common coordinate translation.

To add only the cylinder versions without rerendering existing figures:

```bash
/home/yfjin/Research/anaconda3/envs/graph_analysis/bin/python \
  AnalysisScripts/ManuscriptVisualization/plot_final_load_force_contacts.py \
  --views perspective_3d_cylinder
```

This adds 40 figures (120 files: PDF, PNG, and SVG) and writes a separate
`selection_and_render_manifest__perspective_3d_cylinder.json`, preserving the
original views and their manifest. A full default run exports all four views.
