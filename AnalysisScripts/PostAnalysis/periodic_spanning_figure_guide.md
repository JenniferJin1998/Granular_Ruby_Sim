# Guide to the any-cluster percolation figures

All figures compare 20 simulations at 0° (blue) with 20 at 30° (orange).
The x axis n is a force multiplier, not force itself: each simulation retains
particle–particle contacts with F >= n times its OWN original particle–particle
mean normal force. Wall contacts do not enter this network. A spanning cluster
contains particles contacting both physical bottom and top surfaces.

## The two extent figures: what is the difference?

First flatten the corrugated geometry with the coordinate

`eta_i = [z_i - bottom(x_i)] / [top(x_i) - bottom(x_i)]`.

Here bottom/top are fitted physical wall heights at that particle's horizontal
position. Eta=0 is the bottom surface; eta=1 is the top surface. Eta is
dimensionless and measures relative height, not a graph centrality or angle.

**`largest_cluster_eta_extent.png` (particle centers only):** at each n, find
the strong component containing the most particles, then compute
`max(eta_i) - min(eta_i)` for its particle centers. Average that value over the
20 simulations. Particle centers do not lie on the physical surfaces, so even
a fully spanning network has a value below one (approximately 0.83 here).
For example, centers from eta=0.08 to eta=0.92 give extent=0.84.

**`largest_cluster_boundary_eta_extent.png` (including boundary attachment):**
start with the same cluster and interval. Extend its lower endpoint to zero
if any member contacts the bottom wall, and its upper endpoint to one if any
member contacts the top wall. The example above becomes 1.00 if both walls are
contacted. If only the bottom is contacted, it becomes 0.92. This records known
surface attachment; it does not add wall contacts to the force network.

Both figures follow the **largest strong cluster**, NOT necessarily a spanning
cluster. Its identity can change with n. An average of 0.7 does NOT mean 70% of
simulations span. Only the spanning-probability figure measures that fraction.
Use the center-based measure for geometric coverage and the boundary-inclusive
measure for coverage including attachment. The latter is identically one for
each spanning component, by construction, and is not independent evidence of
spanning. Neither is a physical path length.

## Every figure in `curves/`

| Filename (PNG and SVG) | What the y axis measures |
|---|---|
| `spanning_probability_comparison` | Left: fraction of simulations where ANY component spans. Right: fraction where the largest strong component spans. Shading is a 95% Wilson binomial interval. |
| `spanning_cluster_count` | Number of separate spanning components per simulation, averaged over all 20. Zero is included if none spans. |
| `spanning_particle_count` | Total particles belonging to all spanning components, averaged over all 20. Components are disjoint, so particles are not counted twice. |
| `spanning_particle_fraction` | The preceding total divided by 1,575 particles, averaged over all 20. Not spanning probability. |
| `largest_spanning_cluster_size` | Particle count in the largest SPANNING component. Zero if none spans; the curve averages all 20 simulations. This component may not be the largest strong component overall. |
| `mean_spanning_cluster_size` | First average spanning-component sizes within each simulation. Then average over simulations with at least one spanning component. Simulations without spanning are missing, NOT zero. |
| `spanning_cluster_size_distribution` | Heatmap: distribution of individual spanning-component particle counts at each n. Shared logarithmic size bins; each column sums to one when data exist. All spanning components are pooled, so simulations with more components contribute more. Blank columns mean no spanning components. |
| `largest_cluster_eta_extent` | Center-based local-height extent of the largest strong component, averaged over all 20. See explanation above. |
| `largest_cluster_boundary_eta_extent` | Boundary-inclusive local-height extent of the largest strong component, averaged over all 20. See explanation above. |
| `largest_spanning_eta_extent` | Center-based extent of the largest SPANNING component, averaged only over simulations that span. Its boundary-inclusive counterpart would always equal one. |
| `loading_extent_comparison` | Left: old global-z extent. Right: boundary-inclusive local-height extent. Both concern the largest strong component; lines are means without confidence bands. |

Except for the probability panels and heatmap, shaded curve bands are 95% t
confidence intervals for the mean across contributing simulations, not the
spread of individual particles or a prediction interval. These normal-style
intervals can extend outside physical bounds near 0 or 1. The CSV `count`
column records how many simulations contribute; conditional curves can have
fewer than 20 at high n. No band is available for only one contributing sample.

## Selection and validation figures

| Filename | Meaning |
|---|---|
| `selection/exact_threshold_distributions` | Histogram, boxplots with simulation points, and empirical cumulative distributions of exact ANY-spanning thresholds. These are widest-path thresholds, not rounded grid selections. |
| `selection/old_vs_new_threshold` | Each point is one simulation: x is the old last sampled largest-component spanning threshold; y is the exact any-component spanning threshold. The diagonal indicates equality. A difference can reflect grid rounding and/or a spanning component that was not the largest. |
| `surface_validation/0deg_sim000_surface_fit` | Raw wall-contact points and fitted bottom/top profiles for one flat-geometry simulation. |
| `surface_validation/30deg_sim000_surface_fit` | Same validation for the corrugated geometry. All 40 simulations have numerical fit diagnostics in `tables/surface_fit_audit.csv`. |

## Suggested reading order

1. `spanning_probability_comparison`: does any top-to-bottom connection remain?
2. `exact_threshold_distributions`: when does the final connection disappear?
3. `spanning_cluster_count` and `largest_spanning_cluster_size`: how many paths/
   components remain, and how many particles do they contain? A component can
   contain multiple paths; component count is not independent path count.
4. The two eta-extent figures: how much of the local thickness does the largest
   strong component cover, including versus excluding boundary attachment?

See README.md for full equations, reconstruction assumptions, and output tables.
