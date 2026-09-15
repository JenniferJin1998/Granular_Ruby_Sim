# Analysis results

The three physical-state datasets use the same numbered organization. Start in
the numbered folders rather than in pipeline artifacts.

```text
<dataset>/
├── 0_graph_and_basic_stats/
├── 1_network_property_comparison/
└── 2_force_cluster_comparison/
```

| Section | Purpose |
|---|---|
| `0_graph_and_basic_stats/` | Canonical graph pickle/tables, graph validation, and basic network summaries. |
| `1_network_property_comparison/` | Cross-geometry feature distributions, simulation-mean boxplots, boundary layers, local neighborhoods, and bond order. |
| `2_force_cluster_comparison/` | High-force versus non-high-force comparisons and connected high-force-cluster analysis. |

For final-load and periodic data, force-dependent results are versioned as
`force_split1` and `force_split2`. Split 1 uses twice the pooled 0deg mean over
particle-particle plus wall contacts. Split 2 uses twice the pooled 0deg mean
over particle-particle contacts only. In both versions, the resulting threshold
is applied to the full contact list; connected force clusters contain only
particle-particle contacts. The unversioned legacy result folders are retained
as compatibility locations for split 1.

Within an analysis, use `distributions/` for pooled shapes,
`simulation_mean_boxplots/` for plots where each simulation is one replicate,
`sample_systems/` for representative spatial examples, `relationship_plots/`
for scatter/heatmap summaries, and `tables/` for machine-readable values and
tests. `artifacts/` contains restartable calculations and logs.

## Dataset status

Status below reflects the completed 2026-09-15 regeneration.

| Dataset | Status |
|---|---|
| `final_load/` | Graph, feature-property, bond-order, all four boundary references, and both force-split high/non-high and connected-cluster analyses populated. Local neighborhoods and centered high/non-high analysis remain incomplete. |
| `jamming/` | CSV-reconstructed/validated graph, bond order, bivariate property plots, all four boundary references, and complete-graph final-threshold high/non-high tables populated. Jamming deliberately retains the final-load split-1 threshold; it selects no jamming high-force contacts, so connected force clusters are absent by definition. |
| `periodic_boundaries/2026-08-03/` | Corrected graph, properties, bond order, all three z-surface boundary references, and both force-split high/non-high and connected-cluster analyses populated. Centered-neighborhood Jobs 2/5 are not yet regenerated. |
| `cross_state/` | Preserved unchanged, as requested. |

The periodic folder `_archive_pre_periodic_correction/` retains the complete
historical layout that used the inaccurate inferred periodic lengths. It is
provenance only, not the canonical analysis.

Symlinked presentation files do not duplicate the underlying result data.
