# Community Detection in Social Networks

This project implements and compares different community detection algorithms on social network data, specifically focusing on the Facebook Ego Network dataset.

## Algorithms Implemented

- Louvain Method
- Spectral Clustering
- Label Propagation

## Features

- Network statistics analysis
- Degree distribution visualization
- Community detection and comparison
- Random graph comparison
- Performance metrics (modularity, execution time)

## Requirements

```
networkx
matplotlib
numpy
scikit-learn
python-louvain
scipy
```

## Installation

1. Clone the repository
2. Install requirements:
```bash
pip install networkx matplotlib numpy scikit-learn python-louvain scipy
```

## Usage

Place your network data file (e.g., 'facebook_combined.txt') in the project directory and run:

```bash
python community_detection.py
```

## Output

The script generates:
- Network statistics
- Degree distribution plot (saved as 'degree_distribution.png')
- Community detection results with modularity scores
- Community visualization

## Dataset Format

The input file should be an edge list format where each line contains two node IDs representing an edge:
```
node1_id node2_id
```

## Example Results

- Louvain Method typically achieves modularity ~0.4
- Includes comparison with random graph baseline
- Visualization of detected communities
