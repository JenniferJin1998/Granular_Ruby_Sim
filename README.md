# Granular Ruby Simulation - Graph Analysis

Graph analysis for granular material simulations using Ricci curvature and network analysis.

## Environment Setup

Create and activate the conda environment:
```bash
conda create -n graph_analysis python=3.10
conda activate graph_analysis
pip install numpy scipy networkx pandas matplotlib GraphRicciCurvature
```

## Project Structure

```
Granular_Ruby_Sim/
├── GraphGeneration.py     # Main graph generation and analysis script
├── Data/                  # Data directory (excluded from git)
│   └── Testing/          # Test data
└── README.md             # This file
```

## Usage

Activate the environment and run the script:
```bash
conda activate graph_analysis
python GraphGeneration.py
```

## Requirements

- Python 3.10
- numpy
- scipy
- networkx
- pandas
- matplotlib
- GraphRicciCurvature

## Notes

The `Data/` directory is excluded from version control to avoid tracking large data files.
