from disease_trajectories.shortest_path_network import ShortestPath
from disease_trajectories.pairs_analysis import TemporalPairs

import os
import pandas as pd

if __name__ == "__main__":
    # example dataset for statistical - temporal pairs
    patient_df = pd.read_csv(
        "../example_datasets/example_dataset_pairs_1.csv", delimiter=","
    )
    # statistical pairs object
    pairs_obj = TemporalPairs(patient_df)
    # final pair dataframe
    pairs_df = pairs_obj.final_pairs_df.copy()
    # final trajectory dataframe derived from pairs
    trajectory_df = pairs_obj.trajectories_df_length_3.copy()
    # plot pairs in circular plot
    # pairs_obj.pair_plot(save_fig=True)

    test = 1

    # example dataset for network analysis - shortest path
    # trajectory_df = pd.read_csv("../example_datasets/example_dataset_shortest_path.csv", delimiter=";")
    # network shortest path object
    shortest_path_obj = ShortestPath(trajectory_df)
    # trajectory dataframe with the cluster column, mapping each trajectory to its cluster
    cluster_result_df = shortest_path_obj.perform_clustering(optimal=True)
    # plot network cluster 0
    # shortest_path_obj.cluster_network_plot(
    #     trajectory_df=cluster_result_df,
    #     pairs_df=pairs_df,
    #     cluster_to_plot=0,
    #     percentage_threshold_to_drop=1,
    #     network_radius=11.5,
    #     edge_prc_to_drop=0.01,
    # #   save_fig=True,
    # )

    test = 2
