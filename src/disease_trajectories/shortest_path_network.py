import numpy as np
import pandas as pd
import math
import os

import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from scipy.interpolate import splprep, splev
from matplotlib.patches import FancyArrowPatch, Patch, Circle
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from utils import systems_colors, system_mapping


class ShortestPath:
    """
    A class for clustering trajectory data using network-based shortest path methods.

    This class constructs a graph representation of trajectories, calculates similarity
    measures between diseases based on shortest paths, and performs clustering on the
    resulting similarity matrix.
    """

    def __init__(self, traj_df):
        """
        Initialize the ShortestPathClustering with trajectory data.

        Parameters:
        -----------
        traj_df : pandas.DataFrame
            DataFrame containing trajectory data with conditions as columns
        """
        self.traj_df = traj_df
        self.graph_obj = None
        self.edge_counts = {}
        self.edges = []
        self.traj_len = self.traj_df.shape[1]  # Number of conditions in each trajectory
        self.traj_number = self.traj_df.shape[0]  # Number of trajectories
        self.similarity_matrices = {}
        self.traj_sim_index_df = None
        self.optimal_ncluster = None

        # Initialize the object with required calculations
        self.construct_network_obj()
        self.similarity_matrices_between_trajectories()
        self.traj_similarity_index_matrix()

    def construct_network_obj(self):
        """
        Construct a network graph from trajectory data where nodes are conditions
        and edges represent transitions between conditions with weights based on frequency.
        """
        # Initialize an empty graph
        self.graph_obj = nx.Graph()

        # Extract edges from trajectories (condition pairs in sequence)
        for traj in range(self.traj_number):
            for condition_idx in range(self.traj_len - 1):
                self.edges.append(
                    (
                        self.traj_df.iloc[traj, condition_idx],
                        self.traj_df.iloc[traj, condition_idx + 1],
                    )
                )

        # Count the frequency of edges
        for edge in self.edges:
            if edge in self.edge_counts:
                self.edge_counts[edge] += 1
            else:
                self.edge_counts[edge] = 1

        # Add weighted edges to the graph
        for edge, count in self.edge_counts.items():
            self.graph_obj.add_edge(edge[0], edge[1], weight=count)

    def similarity_index(self, disease_1, disease_2):
        """
        Calculate the similarity index between two diseases using inverse shortest path length.

        Parameters:
        -----------
        disease_1 : str
            First disease name
        disease_2 : str
            Second disease name

        Returns:
        --------
        float
            Similarity index (higher value means more similar)
        """
        try:
            # Calculate shortest path length with weighted edges (inverse of square root of weight)
            shortest_path_length = nx.shortest_path_length(
                self.graph_obj,
                source=disease_1,
                target=disease_2,
                method="dijkstra",
                weight=lambda u, v, d: 1 / np.sqrt(d["weight"]),
            )
            # Convert path length to similarity (inverse relationship)
            similarity = 1 / (
                shortest_path_length + 1
            )  # Adding 1 to avoid division by zero
            return similarity
        except nx.NetworkXNoPath:
            # Return 0 similarity for disconnected diseases
            return 0

    def traj_similarity_matrix(self, traj_indices_1, traj_indices_2):
        """
        Calculate the similarity matrix between diseases in two sets of trajectories.

        Parameters:
        -----------
        traj_indices_1 : list
            List of indices for the first set of trajectories
        traj_indices_2 : list
            List of indices for the second set of trajectories

        Returns:
        --------
        pandas.DataFrame
            Matrix of pairwise disease similarities
        """
        sim_dict = {}
        condition_cols = ["Condition 1", "Condition 2", "Condition 3"]

        # Calculate pairwise similarities between all conditions in both trajectory sets
        for idx_1 in traj_indices_1:
            conditions_1 = self.traj_df.loc[idx_1, condition_cols].tolist()
            for idx_2 in traj_indices_2:
                conditions_2 = self.traj_df.loc[idx_2, condition_cols].tolist()
                for cond_1 in conditions_1:
                    if cond_1 not in sim_dict:
                        sim_dict[cond_1] = {}
                    for cond_2 in conditions_2:
                        sim_dict[cond_1][cond_2] = self.similarity_index(cond_1, cond_2)

        return pd.DataFrame(sim_dict)

    def similarity_matrices_between_trajectories(self):
        """
        Calculate similarity matrices for all pairs of trajectories.
        """
        indices = self.traj_df.index.tolist()

        # Calculate similarity matrix for each pair of trajectories (including self-pairs)
        for i, idx in enumerate(indices):
            # Self similarity (trajectory compared to itself)
            self.similarity_matrices[f"traj_{idx},traj_{idx}"] = (
                self.traj_similarity_matrix([idx], [idx])
            )

            # Cross similarities (trajectory compared to other trajectories)
            for j in range(i + 1, len(indices)):
                jdx = indices[j]
                self.similarity_matrices[f"traj_{idx},traj_{jdx}"] = (
                    self.traj_similarity_matrix([idx], [jdx])
                )

    def traj_similarity_index_matrix(self):
        """
        Create a matrix of average similarity indices between trajectories.
        """
        traj_sim_index_matrix = {}

        # Calculate average similarity for each trajectory pair
        for traj_pair, matrix in self.similarity_matrices.items():
            key1, key2 = traj_pair.split(",")

            # Initialize dictionary keys if needed
            if key1 not in traj_sim_index_matrix:
                traj_sim_index_matrix[key1] = {}
            if key2 not in traj_sim_index_matrix:
                traj_sim_index_matrix[key2] = {}

            # Store average similarity (symmetric matrix)
            avg_similarity = matrix.values.mean()
            traj_sim_index_matrix[key1][key2] = avg_similarity
            traj_sim_index_matrix[key2][key1] = avg_similarity

        # Convert to DataFrame and set diagonal to 1 (self-similarity)
        self.traj_sim_index_df = pd.DataFrame(traj_sim_index_matrix)
        np.fill_diagonal(self.traj_sim_index_df.values, 1)

    def simple_network_plot(self):
        """
        Create a simple visualization of the network graph.
        """
        plt.figure(figsize=(12, 8))

        # Use spring layout for node positioning
        pos = nx.spring_layout(self.graph_obj)

        # Draw nodes, edges, and labels
        nx.draw_networkx_nodes(self.graph_obj, pos, node_size=300, alpha=0.9)
        nx.draw_networkx_edges(self.graph_obj, pos, width=2, alpha=0.5)
        nx.draw_networkx_labels(self.graph_obj, pos, font_size=10)

        plt.title("Network Graph of All Trajectories")
        plt.axis("off")
        plt.show()

    def get_optimal_ncluster(self):
        """
        Determine the optimal number of clusters using silhouette score.

        Returns:
        --------
        int
            Optimal number of clusters
        """
        # Test cluster numbers from 2 to 10
        range_n_clusters = list(range(2, 11))
        silhouette_scores = []

        # Calculate silhouette score for each number of clusters
        for n_clusters in range_n_clusters:
            clusterer = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=10)
            cluster_labels = clusterer.fit_predict(self.traj_sim_index_df.values)
            silhouette_avg = silhouette_score(
                self.traj_sim_index_df.values, cluster_labels
            )
            silhouette_scores.append(silhouette_avg)

        # Get the number of clusters with the highest silhouette score
        optimal_ncluster = range_n_clusters[np.argmax(silhouette_scores)]
        self.optimal_ncluster = optimal_ncluster
        return optimal_ncluster

    def perform_clustering(self, optimal=True, num_clusters=None):
        """
        Perform spectral clustering on the trajectory similarity matrix.

        Parameters:
        -----------
        optimal : bool, default=True
            Whether to use the optimal number of clusters determined by silhouette score
        num_clusters : int, optional
            Specific number of clusters to use (overrides optimal if provided)

        Returns:
        --------
        pandas.DataFrame
            The trajectory DataFrame with cluster assignments added
        """
        # Determine number of clusters
        if num_clusters is None or optimal:
            num_clusters = self.get_optimal_ncluster()

        # Perform spectral clustering with precomputed affinity matrix
        spectral_clustering = SpectralClustering(
            n_clusters=num_clusters, affinity="precomputed", random_state=42
        )
        cluster_labels = spectral_clustering.fit_predict(self.traj_sim_index_df.values)

        # Map cluster labels to trajectories
        cluster_dict = dict(zip(self.traj_sim_index_df.index, cluster_labels))
        # Extract trajectory IDs from index names (format: "traj_X")
        cluster_dict = {int(k.split("_")[1]): v for k, v in cluster_dict.items()}

        # Assign cluster labels to trajectories
        self.traj_df.loc[:, "Cluster"] = self.traj_df.index.map(cluster_dict)
        return self.traj_df

    # Define function to calculate tangent positions for arrow endpoints
    @staticmethod
    def calculate_tangent_end_pos(pos, u, v, condition_percentage_dict, bi_dir=None):
        """
        Calculate tangent positions for arrow endpoints to avoid overlapping with nodes.

        Parameters:
        -----------
        pos : dict
            Dictionary of node positions
        u, v : str
            Source and target node names
        condition_percentage_dict : dict
            Dictionary of condition percentages for node sizing
        bi_dir : bool, default=None
            Whether the edge is bidirectional

        Returns:
        --------
        tuple
            ((x1, y1), (x2, y2)) start and end positions for the arrow
        """
        x1, y1 = pos[u].copy()
        x2, y2 = pos[v].copy()
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / dist, dy / dist  # normalize direction vector

        # Adjust positions based on node size and edge type
        if bi_dir:
            if condition_percentage_dict[v] > 10:  # Larger nodes need more offset
                new_x2 = x2 - dx * 0.3
                new_y2 = y2 - dy * 0.3
                new_x1 = x1 + dx * 0.3
                new_y1 = y1 + dy * 0.3
            else:
                new_x2 = x2 - dx * 0.1
                new_y2 = y2 - dy * 0.1
                new_x1 = x1 + dx * 0.1
                new_y1 = y1 + dy * 0.1
            return (new_x1, new_y1), (new_x2, new_y2)
        else:
            if condition_percentage_dict[v] > 10:
                new_x2 = x2 - dx * 0.3
                new_y2 = y2 - dy * 0.3
            else:
                new_x2 = x2 - dx * 0.1
                new_y2 = y2 - dy * 0.1
            return (x1, y1), (new_x2, new_y2)

    def cluster_network_plot(
        self,
        trajectory_df,
        pairs_df,
        cluster_to_plot,
        percentage_threshold_to_drop=0,
        edge_prc_to_drop=0,
        network_radius=6.5,
        give_node_color_to_edges=False,
        margins=0.2,
        circular_moon=None,
        label_position_y=None,
        save_fig=None,
    ):
        """
        Create a network plot visualization for a specific cluster of disease trajectories.

        This method generates a network graph where nodes represent conditions/diseases
        and edges represent transitions between them in the disease trajectories.
        Nodes are colored by body system and sized by their frequency in the dataset.

        Parameters:
        -----------
        pairs_df : pandas.DataFrame
            DataFrame containing disease pairs information with columns:
            Condition1, Condition2, Num_D1D2

        cluster_to_plot : int
            ID of the cluster to visualize

        percentage_threshold_to_drop : float
            Threshold for dropping conditions with frequency below this percentage

        edge_prc_to_drop : float
            Threshold for dropping edges representing less than this percentage of transitions

        network_radius : float, default=None
            Radius of the network layout (defaults to 8.5 if None)

        circular_moon : int, default=None
            Number of additional points to add to the circular layout

        label_position_y : float, default=None
            Y-offset for node labels (defaults to 0.6 if None)

        give_node_color_to_edges : bool, default=None
            Whether to color edges based on the source node's system color

        margins : float, default=None
            Margins for the network plot

        save_fig : str, default=None
            If provided, save the figure to this filename
        """

        # ---- Data Preparation ----

        # Copy the trajectory dataframe to avoid modifying the original
        traj_df = trajectory_df.copy()

        # Group trajectories by cluster and create separate dataframes
        number_of_clusters_list = traj_df["Cluster"].unique()
        traj_dict = {}
        for i in number_of_clusters_list:
            traj_dict[f"cluster_{i}_df"] = traj_df[traj_df["Cluster"] == i].reset_index(
                drop=True
            )

        # Extract system and trajectory information for the selected cluster
        system_dict = {}
        for system_df_column in [
            "Condition 1 System",
            "Condition 2 System",
            "Condition 3 System",
        ]:
            system_dict[system_df_column] = [
                    system_mapping[disease]
                    for disease in traj_df[system_df_column.split(" System")[0]]
                ]

        system_df = pd.DataFrame(system_dict)

        traj_df = traj_dict[f"cluster_{cluster_to_plot}_df"][
            ["Condition 1", "Condition 2", "Condition 3"]
        ].copy()

        # Standardize condition names (lowercase with spaces instead of underscores)
        for col in ["Condition 1", "Condition 2", "Condition 3"]:
            traj_df[col] = traj_df[col].apply(lambda x: x.replace("_", " ").lower())

        for col in ["Condition1", "Condition2"]:
            pairs_df[col] = pairs_df[col].apply(lambda x: x.replace("_", " ").lower())

        # ---- Network Construction ----

        # Initialize directed graph and edge tracking
        graph_obj = nx.DiGraph()
        edge_counts = {}
        traj_number = traj_df.shape[0]
        traj_len = traj_df.shape[1]

        # Extract edges from trajectories and count occurrences
        edges = []
        node_systems = {}

        for traj in range(traj_number):
            for condition in range(traj_len - 1):
                edges.append(
                    (traj_df.iloc[traj, condition], traj_df.iloc[traj, condition + 1])
                )

        # Count edge frequencies
        for edge in edges:
            if edge in edge_counts:
                edge_counts[edge] += 1
            else:
                edge_counts[edge] = 1

        # Convert pairs dataframe to tuples for easier matching
        pairs_info_list = [
            tuple(x)
            for x in pairs_df[["Condition1", "Condition2", "Num_D1D2"]].to_records(
                index=False
            )
        ]

        # Add weighted edges to the graph with total patient counts
        for edge, count in edge_counts.items():
            for pair_info in pairs_info_list:
                if edge == pair_info[:2]:
                    num_d1d2 = pair_info[-1]
                    graph_obj.add_edge(edge[0], edge[1], weight=count, total=num_d1d2)

        # ---- Identify Bidirectional Edges ----

        bi_directional_edges = set()
        bi_edges_dict = {}
        sum_weights = 0
        total_bi_directional = 0

        for u, v, data in graph_obj.edges(data=True):
            sum_weights += data["weight"]
            if graph_obj.has_edge(v, u):
                bi_directional_edges.add((u, v))
                bi_directional_edges.add((v, u))
                bi_edges_dict[f"{u}_{v}"] = data
                total_bi_directional += data["total"]

        # ---- Calculate Edge Percentages ----

        edge_prc = {}
        total_numd1d2 = 0

        # Calculate total number of patients across all edges
        for u, v, data in graph_obj.edges(data=True):
            # For bidirectional edges, count patients only once (divide by 2)
            total_numd1d2 += (
                data["total"] / 2 if (u, v) in bi_directional_edges else data["total"]
            )

        # Print summary statistics
        print(
            f"The total number of patients included in the pairs that used in trajectories is: {total_numd1d2}\n"
        )
        print(
            f"The total patient of pairs with bidirectional: {total_bi_directional}\n"
        )
        print(
            f"Using the edge percentage {edge_prc_to_drop} the number of patients is {int(total_numd1d2*edge_prc_to_drop)}"
        )

        # Calculate percentage of total patients for each edge
        for u, v, data in graph_obj.edges(data=True):
            edge_prc[f"{u}_{v}"] = data["total"] / total_numd1d2

        # Normalize percentages relative to the maximum
        max_percentage = max(edge_prc.values())

        # ---- Remove Low-Frequency Conditions ----

        # Count conditions across all trajectories
        condition_counts = traj_df.apply(pd.Series.value_counts).sum(axis=1)
        total_traj = traj_df.shape[0]

        # Calculate percentages and create dictionary for node sizing
        condition_percentages = pd.DataFrame((condition_counts / total_traj) * 100)
        condition_percentage_dict = dict((condition_counts / total_traj) * 100)

        # Identify conditions below threshold for removal
        condition_percentages = condition_percentages[
            condition_percentages[0] < percentage_threshold_to_drop
        ]
        condition_to_drop = list(condition_percentages.index)

        # Remove low-frequency nodes from graph
        graph_obj.remove_nodes_from(condition_to_drop)

        # ---- Map Conditions to Systems ----

        for traj in range(traj_number):
            for condition in range(traj_len):
                node_systems.update(
                    {traj_df.iloc[traj, condition]: system_df.iloc[traj, condition]}
                )

        # ---- Assign Colors to Systems ----

        # Use predefined colors from systems_colors dictionary
        systems_by_color = systems_colors.copy()

        # ---- Create Layout for Graph ----

        # Initialize circular layout
        pos = nx.circular_layout(graph_obj)

        # Calculate positions for each system group along circle perimeter
        angs = np.linspace(
            0,
            2 * np.pi,
            (1 if circular_moon is None else circular_moon) + len(systems_by_color),
        )
        repos = []
        radius = 8.5 if network_radius is None else network_radius

        # Create repository of positions
        for ea in angs:
            if ea > 0:
                repos.append(np.array([radius * np.cos(ea), radius * np.sin(ea)]))

        # Adjust positions based on system grouping
        for ea in pos.keys():
            posx = 0
            for i, system_name in enumerate(systems_by_color.keys()):
                if node_systems[ea] == system_name:
                    posx = i
            pos[ea] += repos[posx]

        # ---- Prepare for Visualization ----

        # Create figure
        plt.figure(figsize=(14, 8))

        # ---- Draw Edges ----

        edges = graph_obj.edges(data=True)
        rad = 0.2  # Curve radius for arrow paths
        bi_dir_ploted = []  # Track plotted bidirectional edges
        nodes_kept = []  # Track nodes with edges above threshold

        for u, v, data in edges:
            # Only draw edges above percentage threshold
            if edge_prc[f"{u}_{v}"] >= edge_prc_to_drop:
                nodes_kept.append(u)
                nodes_kept.append(v)

                if not give_node_color_to_edges:
                    # Handle bidirectional edges
                    if (u, v) in bi_directional_edges:
                        if (u, v) not in bi_dir_ploted:
                            bi_dir_ploted.append((v, u))
                            start_pos, end_pos = self.calculate_tangent_end_pos(
                                pos, u, v, condition_percentage_dict, bi_dir=True
                            )
                            # Draw bidirectional arrow
                            edge = FancyArrowPatch(
                                posA=start_pos,
                                posB=end_pos,
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="<|-|>",
                                mutation_scale=14,
                                color="dimgrey",
                                lw=0.01 * data["total"],
                                alpha=1,
                            )
                            plt.gca().add_patch(edge)
                    # Handle unidirectional edges
                    elif (v, u) not in bi_directional_edges:
                        start_pos, end_pos = self.calculate_tangent_end_pos(
                            pos, u, v, condition_percentage_dict, bi_dir=False
                        )
                        edge = FancyArrowPatch(
                            posA=start_pos,
                            posB=end_pos,
                            connectionstyle=f"arc3,rad=-{rad}",
                            arrowstyle="-|>",
                            mutation_scale=14,
                            color="dimgrey",
                            lw=0.01 * data["total"],
                            alpha=1,
                        )
                        plt.gca().add_patch(edge)
                else:
                    # When using node colors for edges
                    start_pos, end_pos = self.calculate_tangent_end_pos(
                        pos, u, v, condition_percentage_dict, bi_dir=False
                    )
                    edge = FancyArrowPatch(
                        posA=start_pos,
                        posB=end_pos,
                        connectionstyle=f"arc3,rad=-{rad}",
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color=(
                            systems_by_color[node_systems[u]]
                            if give_node_color_to_edges
                            else "black"
                        ),
                        lw=110 * data["weight"] / 100,
                        alpha=1,
                    )
                    plt.gca().add_patch(edge)

        # ---- Remove Isolated Nodes ----

        # Remove nodes without edges above threshold
        nodes_to_drop = []
        for node in graph_obj.nodes:
            if node not in nodes_kept:
                nodes_to_drop.append(node)
        try:
            graph_obj.remove_nodes_from(nodes_to_drop)
        except:
            pass

        # ---- Draw Nodes ----

        # Size nodes based on condition frequency
        node_sizes = {
            node: 30 * condition_percentage_dict[node] for node in graph_obj.nodes
        }

        # Draw nodes by system category
        for category, color in systems_by_color.items():
            nodes = [n for n in graph_obj.nodes if node_systems.get(n) == category]
            if not nodes:
                continue

            sizes = [node_sizes[node] for node in nodes]
            nx.draw_networkx_nodes(
                graph_obj,
                pos,
                nodelist=nodes,
                node_size=sizes,
                node_color=color,
                label=f"Category {category}",
                edgecolors="black",
                linewidths=1.2,
                alpha=1,
                margins=margins,
            )

        # ---- Draw Node Labels ----

        # Position labels slightly above nodes
        label_pos = {
            node: (
                pos[node][0],
                pos[node][1] + (0.6 if label_position_y is None else label_position_y),
            )
            for node in graph_obj.nodes
        }

        # Draw labels
        nx.draw_networkx_labels(
            graph_obj,
            label_pos,
            font_size=7.5,
            font_color="black",
            font_family="sans-serif",
            font_weight="light",
        )

        # ---- Draw System Convex Hulls ----

        for category, color in systems_by_color.items():
            nodes = [n for n in graph_obj.nodes if node_systems.get(n) == category]
            if not nodes:
                continue

            points = np.array([pos[n] for n in nodes])

            if len(points) > 2:
                # For 3+ nodes, draw convex hull
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]

                # Calculate centroid and expand hull
                centroid = np.mean(hull_points, axis=0)
                expansion_factor = 3
                expanded_hull_points = hull_points + expansion_factor * (
                    hull_points - centroid
                )

                # Smooth the hull outline
                expanded_hull_points = np.vstack(
                    [expanded_hull_points, expanded_hull_points[0]]
                )
                tck, u = splprep(
                    [expanded_hull_points[:, 0], expanded_hull_points[:, 1]],
                    s=0.0,
                    per=1,
                )
                u_fine = np.linspace(0, 1, 100)
                x_fine, y_fine = splev(u_fine, tck)

                plt.fill(x_fine, y_fine, color=color, alpha=0.3)

            elif len(points) == 2:
                # For exactly 2 nodes, draw circle encompassing both
                midpoint = np.mean(points, axis=0)
                radius = np.linalg.norm((points[0] - points[1]) / 2 + 0.9)
                circle = Circle(
                    midpoint,
                    radius=radius,
                    color=color,
                    alpha=0.3,
                )
                plt.gca().add_patch(circle)

            else:
                # For 1 node, draw circle around it
                for point in points:
                    circle = Circle(point, radius=0.8, color=color, alpha=0.3)
                    plt.gca().add_patch(circle)

        # ---- Add Legend ----

        legend_handles = []
        for system, color in systems_by_color.items():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=system,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=10,
                    alpha=0.6,
                )
            )

        plt.legend(
            handles=legend_handles,
            title="Condition Category",
            loc="upper left",
            fancybox=False,
            shadow=False,
            borderpad=1,
            handletextpad=1,
            handlelength=1.5,
            fontsize="small",
            title_fontsize="small",
        )

        # ---- Finalize Plot ----

        # Add title
        plt.title(
            "Network Visualization of a Cluster with System-Specific Condition Trajectories",
            fontsize=10,
            color="navy",
            fontfamily="serif",
            loc="center",
            pad=20,
        )

        # Save figure if requested
        if save_fig:
            plt.savefig("plot.svg", dpi=100, bbox_inches="tight")

        # Display plot
        plt.show()


if __name__ == "__main__":
    main_directory = os.getcwd()
    print(f"Main directory: {main_directory}")

    # Load example dataset
    trajectory_df = pd.read_csv(
        "../example_datasets/example_dataset_shortest_path.csv", delimiter=","
    )

    # Create shortest path clustering object and perform clustering
    shortest_path_obj = ShortestPath(trajectory_df)
    result_df = shortest_path_obj.perform_clustering(optimal=True)
    print(f"Clustering complete with {shortest_path_obj.optimal_ncluster} clusters")
