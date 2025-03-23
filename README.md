# DECODE-DiseaseTrajectories

## Overview

DECODE-DiseaseTrajectories is a Python package that analyses temporal relationships between disease conditions. It identifies statistically significant disease pairs and determines their temporal directionality through statistical testing. The package then constructs disease trajectories by connecting these significant pairs in sequence and employs a shortest-path graph-based clustering method to group similar disease progression patterns.

This methodology enables researchers to:
- Discover common pathways of disease development
- Visualise condition networks by system category
- Identify clinically meaningful clusters that could inform preventive strategies or treatment approaches

## Installation

### Using pip

```bash
pip install DECODE-DiseaseTrajectories
```

### From source

```bash
git clone https://github.com/raniakous/DECODE-DiseaseTrajectories.git
cd DECODE-DiseaseTrajectories
pip install -e .
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- networkx >= 2.6.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Analysis Workflow

The package implements a complete workflow for disease trajectory analysis:

### 1. Temporal Pairs Analysis

This stage identifies statistically significant pairs of diseases that occur together more frequently than expected by chance:

1. Process patient-level disease occurrence data
2. Identify patients with common conditions
3. Perform Fisher's exact test to identify significant disease pairs
4. Apply Bonferroni correction to control for multiple testing
5. Determine directionality using binomial test
6. Output significant disease pairs with temporal direction

![disease pairs](pairs.png)

### 2. Trajectory Construction

Using the significant pairs from the previous stage:

1. Connect disease pairs to form trajectories
2. Construct trajectories of varying lengths (typically length 3 and 4)
3. Ensure all connections within trajectories are statistically significant
4. Filter out trajectories with repeated conditions

### 3. Network-Based Clustering

Group similar trajectories together using network-based similarity:

1. Construct a network graph from trajectories
2. Calculate similarity between diseases using shortest path distances
3. Generate similarity matrices between trajectories
4. Determine optimal number of clusters using silhouette scores
5. Perform spectral clustering to identify trajectory clusters
![trajectory clusters1](males.pdf)
![trajectory clusters2](females.pdf)

### 4. Visualization

Several visualization methods are available:

1. Disease pair networks (coloured by body system)
2. Cluster-specific network visualizations
3. Interactive network graphs of disease relationships

## Usage Example

```python
from disease_trajectories.pairs_analysis import TemporalPairs
from disease_trajectories.shortest_path_network import ShortestPath
import pandas as pd

# Load patient data
patient_df = pd.read_csv("patient_data.csv")

# Perform temporal pairs analysis
pairs_obj = TemporalPairs(patient_df)

# Get significant disease pairs and trajectories
pairs_df = pairs_obj.final_pairs_df
trajectory_df = pairs_obj.trajectories_df_length_3

# Visualize disease pairs network
pairs_obj.pair_plot()

# Perform network-based clustering
shortest_path_obj = ShortestPath(trajectory_df)
cluster_result_df = shortest_path_obj.perform_clustering(optimal=True)

# Visualize a specific cluster
shortest_path_obj.cluster_network_plot(
    trajectory_df=cluster_result_df,
    pairs_df=pairs_df,
    cluster_to_plot=0,
    percentage_threshold_to_drop=1,
    network_radius=11.5,
    edge_prc_to_drop=0.01,
)
```

## Input Data Format

The package expects longitudinal patient data with the following columns:

- `patient_id`: Unique identifier for each patient
- `date`: Date of disease/condition diagnosis (YYYY-MM-DD format)
- `disease`: Name of the disease/condition

## Output Data

The analysis produces:

- `final_pairs_df`: DataFrame of significant disease pairs with directionality
- `trajectories_df_length_3`: DataFrame of three-condition trajectories
- `trajectories_df_length_4`: DataFrame of four-condition trajectories (if available)
- Cluster assignments for each trajectory

## Citation

If you use this package in your research, please cite:

```
Kousovista, R. (2023). DECODE-DiseaseTrajectories: A Python package for identifying 
and clustering disease progression pathways. Journal of Biomedical Informatics.
```

## License

MIT License

## Contact

For questions or support, please contact [email].
