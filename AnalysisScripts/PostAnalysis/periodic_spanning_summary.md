# Any-cluster percolation and surface-relative loading extent

Start with `FIGURE_GUIDE.md` in the output folder for a plot-by-plot explanation,
including center-based versus boundary-inclusive eta extent.

This is an additive analysis of the corrected periodic 0° and 30° systems
(20 simulations each). Original largest-cluster selection results and figures
are preserved. Run from the project root with the graph_analysis environment:

```bash
python AnalysisScripts/PostAnalysis/periodic_spanning_summary.py
```

Outputs overwrite their previous versions in
`AnalysisResults/periodic_boundaries/2026-08-03/3_force_threshold_percolation/summary/any_cluster_percolation/`.
All figures are saved as PNG and SVG. The existing graph pickle and completed
`summary/tables/sweep_metrics.csv` are required. The script independently
reconstructs all 2,040 threshold networks and checks them against the original
spanning decisions, largest-component sizes, force means, and exact thresholds.

## Selection

Each simulation uses its own original particle–particle mean normal force.
Retain contacts with `F >= n * mean(F_pp)`. Wall nodes and wall contacts never
enter the strong network or the force mean. Bottom/top wall labels only identify
boundary-contacting particles; their wall-contact force is not thresholded.

`selected_n_any_spanning_grid` is the last value in 0:0.1:5 at which ANY
connected component contains both a bottom-contacting and a top-contacting
particle. `first_nonspanning_grid_n` is the next failing sample. An explicit
right-censor flag identifies any system still spanning at the sweep end.

`exact_any_spanning_n` is the primary additional threshold: the maximum, over
all bottom-to-top particle paths, of the minimum normalized contact force on
that path. It is calculated directly by a widest-path algorithm, not by
interpolation. With the inclusive force comparison a path exists at this
threshold and no path exists above it. It is not limited by the 0.1 grid.

The original `selected_n_largest_spanning` is retained for comparison, not
replaced. Unlike largest-component spanning, ANY-component spanning is
monotonic as contacts are removed. Threshold histograms, boxplots with all
simulation points, ECDFs, and old-versus-new scatter plots are in `selection/`.
Tables include geometry means, 95% t intervals, and unadjusted Welch and
Mann–Whitney tests using simulations as replicates.

## Cluster counts and sizes

`tables/component_metrics.csv` records every connected strong component at
every sampled threshold, its particle count, boundary flags, and normalized
extents. IDs are local to a simulation and threshold, not persistent identities.

- `spanning_cluster_count`: number of bottom-to-top components.
- `spanning_particle_count`: total particles in all such components.
- `spanning_particle_fraction`: that total divided by all 1,575 particles.
- `largest_spanning_cluster_size`: maximum spanning-component particle count;
  zero when no component spans.
- `mean_spanning_cluster_size`: arithmetic mean over spanning components within
  each simulation; NaN when none spans. Geometry curves then average these
  simulation means only over simulations that span; `count` in the curve table
  reports the changing number of contributing simulations.

The count, total-size, and largest-size curves include all simulations (zeros
included), with 95% t confidence bands. Conditional size and extent curves use
only available simulations. No missing conditional measurement is filled with
zero. The pooled size-distribution heatmap instead weights each spanning
component equally, using shared logarithmic size bins. Blank columns mean no
spanning clusters, not zero-sized clusters. Probability panels show any-cluster
and largest-cluster spanning fractions with 95% Wilson intervals.

## Loading-direction extent: use local surface-relative height

Dividing cluster z range by global particle z range confounds corrugation with
through-thickness coverage. Even division by one average physical surface gap
does not remove changes in surface elevation across a laterally extended cluster.
Instead define a dimensionless vertical coordinate for each particle:

`eta = (z - z_bottom(x)) / (z_top(x) - z_bottom(x))`.

This uses the local gap measured along loading direction z, NOT perpendicular
distance to a sloped surface. Physical surfaces are reconstructed from the
stored wall `contact_location` coordinates. The current periodic geometries
are extruded in y and piecewise linear in x with breakpoints at quarters of
the exact x period (1.8 mm). Fit four periodic knot heights independently for
bottom and top in each simulation. This geometry-specific assumption must be
revisited for other boundary shapes. It is validated against every stored wall
contact, with a maximum allowed fit residual of 1 micrometre; a failed fit,
crossing surfaces, or particle center outside the surfaces aborts the analysis.

Two complementary component extents are saved:

1. `eta_extent = max(eta) - min(eta)` across particle centers. This is the
   continuous, geometry-normalized extent; it need not reach one even for a
   spanning cluster because particle centers lie inside the physical surfaces.
2. `boundary_eta_extent`: extend the eta interval to 0 if the component contains
   a bottom-contacting particle and to 1 if it contains a top-contacting
   particle. This includes the known boundary attachment and equals one for
   any spanning component. It is NOT an independent test of spanning and does
   not add wall edges to the force network.

Largest-strong-component curves include both definitions and a side-by-side
comparison with the old global-z extent. The center-based extent of the largest
spanning component is conditional on spanning; its boundary-inclusive extent
would trivially equal one and is not plotted as an informative size measure.
Extent is a coverage descriptor, not a path length or mechanical load capacity.

`surface_validation/` shows fitted surfaces and raw contact points for simulation
000 of each geometry. `tables/surface_fit_audit.csv` contains all 40 fits, knot
heights, residuals, and minimum/mean/maximum physical gaps. `validation.json`
records the consistency checks. These fits are approximate reconstructions from
saved contact points, not a claim to have recovered exact simulator wall equations.
