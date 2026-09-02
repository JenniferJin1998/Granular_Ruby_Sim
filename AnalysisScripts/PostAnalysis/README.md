# Post-analysis layout

- `ForceCluster_vf0.ipynb` and `PresentationFigurePlots.ipynb` are the original notebooks.
- `organized_notebooks/01_analysis_draft_workflow.ipynb` is the clean copy for analysis, force-cluster extraction, draft plots, and exploratory calculations.
- `organized_notebooks/02_publication_figure_workflow.ipynb` is the clean copy for polished final plots.
- `organized_notebooks/03_final_vs_jamming_comparison.ipynb` compares final-load and jamming-state graph properties.
- `organized_notebooks/04_jamming_final_threshold_node_groups.ipynb` analyzes corrected jamming-state node properties grouped by final-state high-force node labels.
- `relabel_jamming_with_final_threshold.py` creates the corrected jamming graph list using the final-load `0deg` high-force threshold.
- `compare_load_states.py` writes node-level and mean-change comparison tables.
- `transfer_final_force_node_groups.py` transfers final-state high-force node labels onto jamming node features and writes grouped node-property stats.
