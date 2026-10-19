# Force-cluster elongation and planarity

These supplemental plots use the saved particle-center gyration eigenvalues
for force_split2 connected particle–particle clusters. The threshold remains
twice the pooled 0° particle–particle mean force, not twice each system's mean.
No clusters or old results are changed. Coordinates were periodically unwrapped
in the original calculation. Particle radius is NOT included in these measures.

Write eigenvalues in descending order: lambda1 >= lambda2 >= lambda3 >= 0.
Their square roots are RMS spatial spreads, so ratios of lengths use square
roots of eigenvalue ratios, not the eigenvalue ratios themselves.

## Ratios

- Elongation E = sqrt(lambda1/lambda2): longest/intermediate spread.
- Flatness F = sqrt(lambda2/lambda3): intermediate/thinnest spread.
- Old aspect ratio A = sqrt(lambda1/lambda3) = E*F when both ratios are defined.

The old A is NOT purely planarity: both a long thin chain and a broad thin sheet
can have a large A. F is a useful thickness contrast, but a large F alone also
does not distinguish a sheet from a very elongated, thin ribbon. Read E and F
together, or use the bounded measures below.

## Bounded measures (recommended for planarity)

- Linearity L = (lambda1-lambda2)/lambda1.
- Planarity P = (lambda2-lambda3)/lambda1.
- Sphericity S = lambda3/lambda1.

All lie in [0,1] and L+P+S=1. Here “sphericity” means eigenvalue isotropy, not
surface-area-based sphericity of a solid object.

| Ideal center arrangement | L | P | S |
|---|---:|---:|---:|
| Straight line | 1 | 0 | 0 |
| Isotropic sheet in a plane | 0 | 1 | 0 |
| Isotropic 3D cloud | 0 | 0 | 1 |

A planar but very elongated cluster can have high L and low P: these descriptors
separate sheet-like geometry from line-like geometry rather than just testing
whether all points lie in a plane. All are center-cloud descriptors, not the
physical thickness of finite-sized particles.

## Degenerate clusters and weighting

Two centers are necessarily collinear; three centers are necessarily planar.
Such clusters should not be interpreted as evidence of emergent chain/sheet
organization. Every distribution is provided both for ALL clusters and for
clusters with at least four particles. Four is a minimum for noncoplanar centers,
not a guarantee of reliable shape estimation.

Ratios with denominator <= 1e-10*lambda1 are stored as NaN and omitted from
ratio histograms (never replaced by huge finite numbers). The legend and table
report finite and total counts. The relative tolerance is scale-independent;
it differs from the old job's absolute numerical cutoff. Bounded measures remain
defined for nonzero-spread line/plane clusters. `spatial_rank` counts eigenvalues
above this relative tolerance. Values are derived from saved eigenvalues, not a
new unwrapping or recomputation of the clusters.

Each cluster has equal weight in pooled histograms; this is not particle-weighted
or simulation-equal weighting, and these plots alone are not significance tests.
The ratio plots use logarithmic x bins and show probability MASS per bin, not
density per unit x. The bounded plots use shared linear bins. No long-tail values
are hidden by an upper cutoff. The separate simulation-mean table uses only
finite values for each metric; it is distinct from the pooled histograms.

## Outputs and regeneration

- `elongation_and_flatness_*.png/svg`: E and F distributions, all / >=4 particles.
- `bounded_shape_measures_*.png/svg`: L, P and S distributions, same two subsets.
- `linearity_vs_planarity.png/svg`: per-cluster L/P values for >=4 particles,
  colored by log10 particle count; each point lies below L+P=1.
- `cluster_shape_metrics.csv`: original rows plus the six new diagnostic fields.
- `distribution_summary.csv`: inclusion/exclusion counts and descriptive statistics.
- `simulation_mean_shape_metrics.csv`: finite-value means within each simulation.

Run `python AnalysisScripts/PostAnalysis/force_cluster_shape_diagnostics.py`.
Outputs are saved under the existing connected-force-cluster folder in
`shape_diagnostics/`; reruns overwrite these supplementary files only.
