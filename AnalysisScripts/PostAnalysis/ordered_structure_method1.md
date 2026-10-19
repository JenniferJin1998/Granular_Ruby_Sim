# Method 1 — crystal-reference feature similarity

## Purpose and limits

Locate particles whose **contact-network environments** resemble the existing
ideal SC/BCC/FCC/HCP references, then inspect their neighboring regions.
This is an exploratory candidate-ranking method, not a calibrated classifier
of crystalline phases. Every eligible particle has a nearest reference even
when it is far from ALL crystals. A top-ranked particle is not necessarily
crystalline, and a high cosine is not a crystallization probability.

Run from the repository root in the graph_analysis environment:

```bash
python AnalysisScripts/PostAnalysis/ordered_structure_method1.py
```

Outputs go to
`AnalysisResults/periodic_boundaries/2026-08-03/4_ordered_structure_detection/method_1_feature_similarity/`.
Reruns overwrite this method's outputs, not the previous periodic analyses.

## Input and reference construction

- Ruby particle properties: corrected periodic `0_graph_and_basic_stats/graph_data/node_features.csv`.
- Ruby bond order: `1_network_property_comparison/bond_order/tables/bond_order_nodes.csv`.
- Ideal crystal particle properties: `AnalysisResults/crystal_references/0_graph_and_basic_stats/graph_data/node_features.csv`.
- Graphs for neighborhoods: the corresponding canonical `graph_dict_labeled.pkl` files.

Join Ruby rows on geometry, simulation ID, and particle ID, validate a one-to-one
match and equality of degree and bond-order coordination. All 63,000 particles
are exported. Wall nodes/edges are excluded from features and neighborhoods.
Wall labels only identify boundary-distance diagnostics. Eligibility requires
degree >=4 and finite features; excluded particles remain in tables with missing
scores and false selection flags. Four is an explicit low-coordination screen,
not a universal physical threshold.

For each crystal, average feature vectors over particles at least two contact
hops from either z surface and with the maximum coordination in that crystal.
This avoids mixing incomplete surface shells into the reference prototype.
The reference degree values are SC=6, BCC=8, FCC=12, HCP=12 under the EXISTING
contact-neighbor definitions. These are not necessarily the values obtained
with other cutoff/neighbor conventions (particularly BCC).

Degree and local bond order are nearly constant at equivalent bulk crystal
sites. **Closeness and betweenness are NOT universal crystal constants:** they
depend on particle location, sample size, shape, boundaries, and global graph
topology. Reference variation is exported rather than assumed zero. The
ideal systems have similar but not identical sizes/shapes to the Ruby systems.

## Three feature vectors

| Model | Features | Role |
|---|---|---|
| `local` | degree, q4, q6, qbar4, qbar6 | Main local-order screening model; includes coordination and neighbor-averaged bond order. |
| `angular_only` | q4, q6, qbar4, qbar6 | Sensitivity to including coordination. |
| `network_augmented` | degree, closeness, betweenness, q4, q6 | The proposed network-feature vector; exploratory sensitivity only because centralities are global and boundary-dependent. |

qbar4/qbar6 are the saved neighbor-averaged bond-order parameters, not arithmetic
averages of scalar q4/q6. They add information about similarity of neighboring
bond orientations. See Lechner and Dellago's original method:
https://arxiv.org/abs/0806.3345 . The current contact graph is used consistently;
missing physical contacts can make a geometrically ordered environment appear
unlike a fully coordinated ideal crystal. A future geometric-neighbor analysis
should therefore be kept separate from this contact-based screen.

## Scaling and scores

For feature j use a common scale shared by every Ruby particle and lattice:

`s_j = sqrt(mean_over_four_crystal_prototypes(reference_j**2))`.

This uncentered scaling puts degree, centralities, and q features on comparable
reference-relative scales without dividing by near-zero ideal-crystal variance.
It gives equal feature weights after scaling, not statistically optimal weights.
Feature correlations are not removed; the alternative feature models expose
some sensitivity to this choice. Do not compare numerical distances across
different feature models as if their definitions were identical.

For scaled particle vector u and reference vector v:

- **Cosine** = `(u dot v) / (norm(u)*norm(v))`. Higher is more similar in feature
  direction. `feature_angle_degrees = acos(maximum_cosine)` is an angle in
  FEATURE SPACE, not a physical contact/bond angle.
- **Scaled RMS distance** = `sqrt(mean((u-v)**2))`. Lower is a closer match in
  both relative feature values and magnitudes. This is the recommended main
  ranking alongside cosine, not a probabilistic crystallinity score.

Cosine alone has a fundamental ambiguity: a vector twice the reference vector
has cosine 1 despite different property values. The script explicitly verifies
this counterexample and also exports unscaled/raw cosines to expose scale
dominance. With degree near 6–12 and betweenness near 0.005, a raw dot product
would heavily favor degree. Formula reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html .

Each particle is compared to all four lattices; both best-reference names and
all four distances/cosines are exported. The cosine-best and distance-best
reference may differ. `distance_margin` is the difference between the two best
distances, not a confidence probability; a small margin signals ambiguity.

## Candidate selection and spatial checks

For each model, the pooled lowest 5% of distances among eligible particles is
a common relative screen across BOTH geometries. The highest 5% of cosine
values is also saved. These are visualization/ranking cutoffs only, NOT
crystalline phase thresholds. Do not interpret a 5% screen as 5% crystallinity.
Ties can make the selected proportion slightly larger than 5%.

No top fraction is imposed separately on each simulation or geometry. Exported
pooled and within-system ranks let users change the visualization selection.
Population comparisons use simulation-level summaries (20 replicates each),
not independent-particle significance tests.

For local-distance candidates, connected regions in the original contact graph
are tabulated. Neighbor candidate fraction, neighbor agreement on nearest
reference, and shortest contact-hop distance to a wall are also exported.
`mean_s6` and `fraction_s6_above_threshold` are held OUT of the feature vector:
they provide complementary bond-coherence checks in the selected population.
These are not statistically independent validation data because they come from
the same configurations. Boundary-distance NaN means no path to a boundary,
not a bulk particle. Short boundary distances identify possible surface bias.

## Figure guide

| Figure(s) | What they show |
|---|---|
| `figures/score_distributions_and_cosine_caution` | Pooled eligible-particle distance distributions, scaled cosine distributions, and cosine versus distance (fixed random subsample for readability). A high cosine alone can still have substantial distance. |
| `figures/q4_q6_with_bulk_references` | Ruby q4/q6 points colored by the full LOCAL feature distance, with ideal bulk prototypes marked. This 2D projection omits degree/qbar features and is not itself the classifier. |
| `figures/simulation_level_comparison` | Each dot is one simulation: mean eligible-particle distance and fraction of ALL 1,575 particles passing the common pooled top-5% distance screen. |
| `figures/*_representative_spatial_map` | x–z and y–z projections of the simulation containing the best local-distance particle in each geometry. Color is distance; red outlines mark screened particles; gray particles are unscored. These are representative inspections, not all-system averages. |
| `neighborhoods/*_distance_*` | Two-hop neighborhood of a strong distance candidate, ideal reference neighborhood, and scaled feature profile. Three candidates per geometry are chosen from distinct simulations; displayed rank is rank among these distinct-simulation representatives. |
| `neighborhoods/*_cosine_*` | Same inspection for cosine-ranked candidates, allowing direct checks of cosine-only false similarity. |

Neighborhoods use minimum-image periodic unwrapping, particle–particle contacts
only, equal spatial scales within each 3D panel, and micrometre coordinates.
The center is a red star. Reference and Ruby neighborhoods retain their original
orientations; they are NOT aligned by a best-fit rotation. Different appearance
alone is not proof of a structural mismatch. Two-hop neighborhoods expose shell
coherence, but this method does not perform geometric template fitting.
PNG and SVG versions are provided for every figure.
Ideal reference particles are gray; their color is not a calculated similarity
score. Ruby neighborhood colors share a 0–0.8 distance range (larger values
saturate); unscored Ruby particles are light gray.

## Table guide

- `*_particle_scores.csv`: all particles, source features, eligibility, scores
  to all four references, best matches, ranks and candidate flags. The local
  table additionally contains boundary and neighbor-coherence diagnostics.
- `*_reference_prototypes.csv`, `*_reference_variation.csv`, `*_feature_scales.csv`:
  exact prototypes, within-reference variation and reproducible scaling.
- `*_screening_cutoffs.json`: numerical pooled relative-screen thresholds.
- `simulation_summaries.csv`: replicate-level means and screened fractions.
- `model_sensitivity.csv`: rank correlations and candidate-set overlap between
  feature choices, plus agreement of cosine and distance rankings.
- `independent_order_and_boundary_checks.csv`: held-out bond-coherence and
  boundary descriptors for eligible particles and both candidate populations,
  separately in every simulation.
- `top5_connected_regions.csv`: spatially connected candidate components and
  member IDs; adjacency does not imply identical crystal-reference assignment.
- `visualized_candidates.csv`, `visualized_neighborhood_members.csv`: particle
  IDs needed to reproduce each selected region.
- `validation.json`: join/feature consistency and numerical score checks.

## Sensible next methods

1. Check that ranked regions also have coherent bond orientations (s6), not just
   similar scalar descriptors; inspect boundary bias and model sensitivity.
2. Calibrate against perturbed ideal crystals and genuinely disordered controls
   with comparable density, coordination and neighbor definitions before setting
   an absolute ordered/disordered threshold.
3. Use rotation-invariant geometric neighbor-shell/template matching as a
   separate method, ideally with geometric neighbors rather than only loaded
   contacts. This can test actual packing similarity beyond scalar features.

Until then, report these as **crystal-reference-like contact environments**,
not detected SC/BCC/FCC/HCP phases or measured crystalline fractions.
