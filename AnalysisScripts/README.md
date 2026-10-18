# AnalysisScripts layout

- `Pipeline/`: staged graph-building pipeline code (`GraphPipeline*.py`).
- `RunWhole/`: monolithic graph-generation script and its reference notes.
- `PostAnalysis/`: notebooks and visualization helpers used after graphs are built.
- `ManuscriptVisualization/`: finalized manuscript plotting code; figure exports live under each case's `AnalysisResults/<case>/Visualization/`.
- `jobs/`: SLURM submission scripts for both pipeline and whole-run workflows.
- `OldVersions/`: archived earlier scripts.
