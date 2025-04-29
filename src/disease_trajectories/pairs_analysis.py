import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy.stats import fisher_exact, binomtest

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from utils import systems_colors, system_mapping


class TemporalPairs:
    """
    A class for analyzing temporal relationships between disease pairs.

    This class identifies patient pairs with common conditions, performs statistical
    tests to determine significant disease pairs, and constructs disease trajectories.
    """

    def __init__(self, condition_df):
        """
        Initialize the TemporalPairs analysis with condition data.

        Parameters:
        -----------
        condition_df : pandas.DataFrame
            DataFrame containing patient condition data with columns:
            patient_id, disease, date
        """
        # Keep only the first occurrence of each disease for each patient
        self.condition_df = condition_df.drop_duplicates(
            subset=["patient_id", "disease"], keep="first"
        ).reset_index(drop=True)

        # Set up directory for intermediate text files
        self.txt_directory = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "txt_files",
        )

        # Create txt_files directory if it doesn't exist
        os.makedirs(self.txt_directory, exist_ok=True)

        # Initialize data structures
        self.disease_pairs_df2 = None
        self.fisher_stats = None
        self.filtered_stats = None
        self.binomial_stats = None
        self.final_pairs_df = None
        self.trajectories_df_length_3 = None
        self.trajectories_df_length_4 = None

        # Run analysis pipeline
        self.extract_patient_pairs_txt()
        self.disease_pairs_counter()
        self.read_extracted_condition_2_txt()
        self.perform_fisher_test()
        self.perform_binomial_test()
        self.get_trajectories()

    def extract_patient_pairs_txt(self):
        """
        Extract patient pairs with common conditions and write to text files.

        Creates a text file for each number of common conditions, with each line
        containing information about a patient pair.
        """
        # Dictionaries to store patient data
        all_patients = {}  # Maps patient_id to list of conditions
        all_patient_disease_dates = {}  # Maps patient_id to list of dates

        # Process each row in the data
        for _, row in self.condition_df.iterrows():
            patient_id = str(row["patient_id"])
            date = row["date"]
            condition = row["disease"]

            # Initialize patient data if not already present
            if patient_id not in all_patients:
                all_patients[patient_id] = []
                all_patient_disease_dates[patient_id] = []

            # Add condition and date to patient data
            all_patients[patient_id].append(condition)
            all_patient_disease_dates[patient_id].append(date)

        # Dictionary to hold output file handles
        output_files = {}

        # Loop through each pair of patients
        patient_ids = list(all_patients.keys())
        for i, id1 in enumerate(patient_ids):
            for id2 in patient_ids[i + 1 :]:
                # Find common diseases between patients
                common_diseases = set(all_patients[id1]).intersection(
                    set(all_patients[id2])
                )

                # Skip if fewer than 2 common diseases
                if len(common_diseases) < 2:
                    continue

                # Create output file if not already created
                if len(common_diseases) not in output_files:
                    output_files[len(common_diseases)] = open(
                        os.path.join(
                            self.txt_directory, f"traj_{len(common_diseases)}.txt"
                        ),
                        "w",
                    )

                # Write patient pair information to file
                p1_common_diseases = [
                    d for d in all_patients[id1] if d in common_diseases
                ]
                p2_common_diseases = [
                    d for d in all_patients[id2] if d in common_diseases
                ]
                p1_dates = [
                    all_patient_disease_dates[id1][i]
                    for i, d in enumerate(all_patients[id1])
                    if d in common_diseases
                ]
                p2_dates = [
                    all_patient_disease_dates[id2][i]
                    for i, d in enumerate(all_patients[id2])
                    if d in common_diseases
                ]

                output_files[len(common_diseases)].write(
                    "{}\n".format(
                        "\t".join(
                            [
                                id1,
                                id2,
                                str(len(common_diseases)),
                                "\t".join(p1_common_diseases),
                                "\t".join(p2_common_diseases),
                                "\t".join(p1_dates),
                                "\t".join(p2_dates),
                            ]
                        )
                    )
                )

        # Close all output files
        for output_file in output_files.values():
            output_file.close()

    def change_date_format(self):
        """
        Process traj_2.txt file to prepare for disease pairs counting.
        """
        traj2_path = os.path.join(self.txt_directory, "traj_2.txt")
        traj2b_path = os.path.join(self.txt_directory, "traj_2b.txt")

        # Read traj_2.txt
        try:
            with open(traj2_path, "r") as file:
                lines = file.readlines()

            # Write to traj_2b.txt
            with open(traj2b_path, "w") as out_file:
                for line in lines:
                    out_file.write(line)

        except FileNotFoundError:
            print(f"Warning: File {traj2_path} not found.")

    @staticmethod
    def add_one_dictionary(dictionary, key):
        """
        Increment a counter in a dictionary.

        Parameters:
        -----------
        dictionary : dict
            Dictionary containing counters
        key : hashable
            Key to increment

        Returns:
        --------
        dict
            Updated dictionary
        """
        dictionary[key] = dictionary.get(key, 0) + 1
        return dictionary

    @staticmethod
    def calculate_stats(periods):
        """
        Calculate mean and standard deviation of time periods.

        Parameters:
        -----------
        periods : list
            List of time periods (in days)

        Returns:
        --------
        tuple
            (mean, standard deviation)
        """
        if periods:
            avg = np.mean(periods)
            std_dev = np.std(periods)
        else:
            avg = "NaN"
            std_dev = "NaN"
        return avg, std_dev

    def disease_pairs_counter(self):
        """
        Count disease pairs and calculate statistics for each pair.

        Processes traj_2b.txt to count how many patients have each disease pair
        and calculates the average time between diseases.
        """
        self.change_date_format()

        # Define input and output file paths
        input_path = os.path.join(self.txt_directory, "traj_2b.txt")
        output_path = os.path.join(self.txt_directory, "conditions_2b.txt")

        # Data structures to store pair information
        periods_dict = {}  # Maps disease pair to list of time periods
        patients_dict = {}  # Maps disease pair to set of patient IDs

        try:
            with open(input_path, "r") as f:
                for line in f:
                    fields = line.strip().split("\t")

                    # Extract patient IDs, diseases, and dates
                    pat1 = fields[0]
                    pat2 = fields[1]

                    dis11 = fields[3]
                    dis12 = fields[4]
                    dis21 = fields[5]
                    dis22 = fields[6]

                    try:
                        date11 = datetime.strptime(fields[7], "%Y-%m-%d")
                        date12 = datetime.strptime(fields[8], "%Y-%m-%d")
                        date21 = datetime.strptime(fields[9], "%Y-%m-%d")
                        date22 = datetime.strptime(fields[10], "%Y-%m-%d")
                    except (ValueError, IndexError):
                        print(f"Warning: Date format issue in line: {line}")
                        continue

                    # Create disease pairs
                    pair1 = f"{dis11}\t{dis12}"
                    pair2 = f"{dis21}\t{dis22}"

                    # Calculate time periods
                    period1 = (date12 - date11).days
                    period2 = (date22 - date21).days

                    # Add periods to dictionary
                    if pair1 not in periods_dict:
                        periods_dict[pair1] = []
                    periods_dict[pair1].append(period1)

                    if pair2 not in periods_dict:
                        periods_dict[pair2] = []
                    periods_dict[pair2].append(period2)

                    # Add patients to dictionary
                    patients_dict.setdefault(pair1, set()).add(pat1)
                    patients_dict.setdefault(pair2, set()).add(pat2)

            # Write statistics to output file
            with open(output_path, "w") as f:
                for pair, periods in periods_dict.items():
                    unique_pat = len(patients_dict[pair])
                    avg, std_dev = self.calculate_stats(periods)
                    f.write(f"{pair}\t{unique_pat}\t{avg}\t{std_dev}\n")

        except FileNotFoundError:
            print(f"Warning: File {input_path} not found.")

    def read_extracted_condition_2_txt(self):
        """
        Read and filter the disease pairs file.

        Loads the disease pairs from conditions_2b.txt and filters out pairs
        with fewer than 10 patients.
        """
        conditions_path = os.path.join(self.txt_directory, "conditions_2b.txt")

        try:
            # Load the disease pairs
            disease_pairs_df = pd.read_csv(conditions_path, sep="\t", header=None)

            # Assign column names
            disease_pairs_df.columns = [
                "Condition1",
                "Condition2",
                "NPatients",
                "AvgDays",
                "StdDev",
            ]

            # Filter for pairs with at least 10 patients
            self.disease_pairs_df2 = disease_pairs_df[
                disease_pairs_df["NPatients"] >= 10
            ]

        except FileNotFoundError:
            print(f"Warning: File {conditions_path} not found.")
            self.disease_pairs_df2 = pd.DataFrame(
                columns=["Condition1", "Condition2", "NPatients", "AvgDays", "StdDev"]
            )

    def perform_fisher_test(self):
        """
        Perform Fisher's exact test on disease pairs.

        Tests whether each disease pair co-occurs more frequently than expected
        by chance.
        """
        # Skip if no disease pairs are available
        if self.disease_pairs_df2.empty:
            self.fisher_stats = pd.DataFrame()
            return

        # Convert disease pairs to list of tuples
        disease_pairs = list(self.disease_pairs_df2.itertuples(index=False, name=None))

        # Get sets of diseases for each patient
        all_patients = (
            self.condition_df.groupby("patient_id")["disease"].apply(set).tolist()
        )

        result = []
        # Perform Fisher's exact test for each disease pair
        for dis1, dis2, npat, avg_days, std_dev in disease_pairs:
            # Count patients in each category
            num_d1d2 = 0  # Has both diseases
            num_d1 = 0  # Has only disease 1
            num_d2 = 0  # Has only disease 2
            num_not = 0  # Has neither disease

            for patient in all_patients:
                has_d1 = dis1 in patient
                has_d2 = dis2 in patient

                if not has_d1 and not has_d2:
                    num_not += 1
                elif not has_d1 and has_d2:
                    num_d2 += 1
                elif has_d1 and not has_d2:
                    num_d1 += 1
                else:  # has both
                    num_d1d2 += 1

            # Perform Fisher's exact test
            contingency_table = [[num_d1d2, num_d1], [num_d2, num_not]]
            _, p_value = fisher_exact(contingency_table)

            # Store results
            result.append(
                [
                    dis1,
                    dis2,
                    npat,
                    avg_days,
                    std_dev,
                    p_value,
                    num_d1d2,
                    num_d1,
                    num_d2,
                    num_not,
                ]
            )

        # Convert results to DataFrame
        self.fisher_stats = pd.DataFrame(
            result,
            columns=[
                "Condition1",
                "Condition2",
                "NPatients",
                "AvgDays",
                "StdDev",
                "Fisher_p_value",
                "Num_D1D2",
                "Num_D1",
                "Num_D2",
                "Num_Not",
            ],
        )

    def perform_binomial_test(self, p_value_threshold=True):
        """
        Perform binomial test and Bonferroni correction on disease pairs.

        Identifies significant disease pairs and determines directionality.
        """
        # self.fisher_stats["Fisher_p_value"] = self.fisher_stats["Fisher_p_value"].apply(
        #     lambda x: x / 10000
        # )
        # Skip if Fisher stats are not available
        if not hasattr(self, "fisher_stats") or self.fisher_stats.empty:
            self.filtered_stats = pd.DataFrame()
            self.binomial_stats = pd.DataFrame()
            self.final_pairs_df = pd.DataFrame()
            return

        # Copy Fisher stats
        stats = self.fisher_stats.copy()

        # Create a ranking for disease pairs
        stats["rank"] = 0
        rank = 1
        i = 0

        while i < len(stats):
            if stats.loc[i, "rank"] != 0:
                i += 1
                continue

            # Find reverse pairs (condition1 and condition2 swapped)
            reverse_condition = (stats["Condition1"] == stats.loc[i, "Condition2"]) & (
                stats["Condition2"] == stats.loc[i, "Condition1"]
            )
            reverse_indices = stats.index[reverse_condition].tolist()

            if len(reverse_indices) == 1:
                # Assign same rank to pair and its reverse
                stats.loc[i, "rank"] = rank
                stats.loc[reverse_indices[0], "rank"] = rank
                rank += 1
            elif len(reverse_indices) == 0:
                # Assign new rank to pair without a reverse
                stats.loc[i, "rank"] = rank
                rank += 1

            i += 1

        if p_value_threshold:
            # Perform Bonferroni correction
            num_unique_ranks = len(stats["rank"].unique())
            p_value_threshold = 0.001  # highly statistically significant
            corrected_threshold = p_value_threshold / num_unique_ranks

            # Filter for significant pairs
            self.filtered_stats = stats[
                stats["Fisher_p_value"] < corrected_threshold
            ].reset_index(drop=True)
        else:
            self.filtered_stats = stats.copy()

        if self.filtered_stats.empty:
            self.binomial_stats = pd.DataFrame()
            self.final_pairs_df = pd.DataFrame()
            return

        # Initialize directions (3 = neutral, 1 = forward, -1 = reverse, 0 = bidirectional)
        directions = [3] * len(self.filtered_stats)

        # Determine directionality for each pair
        for i, row in self.filtered_stats.iterrows():
            # Find pairs with the same rank
            same_rank_indices = self.filtered_stats[
                self.filtered_stats["rank"] == row["rank"]
            ].index
            same_rank_rows = self.filtered_stats.loc[same_rank_indices]

            if len(same_rank_rows) == 2:
                # Compare patient counts for directionality
                n1 = same_rank_rows["NPatients"].iloc[0]
                n2 = same_rank_rows["NPatients"].iloc[1]

                # Perform binomial test
                p_value = binomtest(k=n1, n=n1 + n2, p=0.5, alternative="two-sided")

                if p_value.pvalue < 0.01:
                    if n1 > n2:
                        directions[same_rank_indices[0]] = 1
                        directions[same_rank_indices[1]] = -1
                    else:
                        directions[same_rank_indices[1]] = 1
                        directions[same_rank_indices[0]] = -1
                else:
                    directions[same_rank_indices[0]] = 0
                    directions[same_rank_indices[1]] = 0

        # Add direction to filtered stats
        self.filtered_stats["direction"] = directions
        self.binomial_stats = self.filtered_stats

        # Filter for forward and bidirectional pairs
        self.final_pairs_df = (
            self.filtered_stats[self.filtered_stats["direction"].isin([0, 1, 3])]
            .sort_values(by="NPatients", ascending=False)
            .reset_index(drop=True)
        )

    def get_trajectories(self, max_length=4):
        """
        Generate disease trajectories from significant pairs.

        Parameters:
        -----------
        max_length : int, default=4
            Maximum length of trajectories to generate
        """
        # Skip if no final pairs are available
        if not hasattr(self, "final_pairs_df") or self.final_pairs_df.empty:
            self.trajectories_df_length_3 = pd.DataFrame()
            self.trajectories_df_length_4 = pd.DataFrame()
            return

        # Extract disease pairs
        pairs = list(
            zip(self.final_pairs_df["Condition1"], self.final_pairs_df["Condition2"])
        )
        pair_set = set(pairs)

        # Initialize with length-2 trajectories (pairs)
        trajectories = []
        for pair in pairs:
            trajectories.append((pair[0], pair[1]))

        # Build longer trajectories
        for length in range(3, max_length + 1):
            new_trajectories = []
            for trajectory in trajectories:
                if len(trajectory) == length - 1:
                    last_event = trajectory[-1]
                    for pair in pair_set:
                        if pair[0] == last_event:
                            # Create new trajectory by appending second disease from matching pair
                            new_trajectory = trajectory + (pair[1],)
                            # Check for uniqueness (no repeated conditions)
                            if len(set(new_trajectory)) == len(new_trajectory):
                                new_trajectories.append(new_trajectory)

            trajectories.extend(new_trajectories)

        # Create DataFrame for length 3 trajectories
        length_3_trajectories = [t for t in trajectories if len(t) == 3]
        if length_3_trajectories:
            self.trajectories_df_length_3 = pd.DataFrame(
                length_3_trajectories,
                columns=["Condition 1", "Condition 2", "Condition 3"],
            )
            self.trajectories_df_length_3.insert(0, "Length", 3)
        else:
            self.trajectories_df_length_3 = pd.DataFrame(
                columns=["Length", "Condition 1", "Condition 2", "Condition 3"]
            )

        # Create DataFrame for length 4 trajectories
        length_4_trajectories = [t for t in trajectories if len(t) == 4]
        if length_4_trajectories:
            self.trajectories_df_length_4 = pd.DataFrame(
                length_4_trajectories,
                columns=["Condition 1", "Condition 2", "Condition 3", "Condition 4"],
            )
            self.trajectories_df_length_4.insert(0, "Length", 4)
        else:
            self.trajectories_df_length_4 = pd.DataFrame(
                columns=[
                    "Length",
                    "Condition 1",
                    "Condition 2",
                    "Condition 3",
                    "Condition 4",
                ]
            )

    def calculate_condition_counts(self):
        """
        Calculate the counts and percentages of unique patients for each condition.

        This method counts how many unique patients have each condition and
        calculates the percentage relative to the total number of unique patients.
        The results are stored in the condition_counts_df attribute.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with condition counts and percentages
        """
        # Get total number of unique patients
        total_patients = self.condition_df["patient_id"].nunique()

        # Count unique patients for each condition
        condition_counts = self.condition_df.groupby("disease")["patient_id"].nunique()

        # Convert to DataFrame
        counts_df = condition_counts.reset_index()
        counts_df.columns = ["Condition", "N"]

        # Calculate percentage
        counts_df["N%"] = (counts_df["N"] / total_patients * 100).round(2)

        # Sort by count in descending order
        counts_df = counts_df.sort_values("N", ascending=False).reset_index(drop=True)

        # Store the result
        self.condition_counts_df = counts_df

        return counts_df

    def pair_plot(
        self,
        change_node_size=800,
        node_label_position=0.1,
        reduce_edge_width_by=150,
        save_fig=False,
    ):
        """
        Generate a network visualization of disease pairs with system-based coloring.
        """
        systems_by_color = systems_colors.copy()
        system_mapping_dict = system_mapping
        freq_df = self.calculate_condition_counts()

        condition_frequencies_dict = {
            cond_name.replace("_", " ").lower(): float(cond_percentage)
            for cond_name, cond_percentage in zip(freq_df["Condition"], freq_df["N%"])
        }

        max_size = max(condition_frequencies_dict.values())
        node_sizes = {
            cond_name: change_node_size * (cond_percentage / max_size) ** 2
            for cond_name, cond_percentage in condition_frequencies_dict.items()
        }

        pairs_df = self.final_pairs_df.copy()
        pairs_df["Condition1"] = pairs_df["Condition1"].apply(
            lambda x: x.replace("_", " ").lower()
        )
        pairs_df["Condition2"] = pairs_df["Condition2"].apply(
            lambda x: x.replace("_", " ").lower()
        )

        missing_conditions = set(pairs_df["Condition1"]) - set(
            system_mapping_dict.keys()
        )
        if missing_conditions:
            raise ValueError(
                f"Missing conditions in system mapping: {missing_conditions}."
            )

        pairs_df["system1"] = pairs_df["Condition1"].apply(
            lambda x: system_mapping_dict.get(x, "Unknown")
        )
        pairs_df["system2"] = pairs_df["Condition2"].apply(
            lambda x: system_mapping_dict.get(x, "Unknown")
        )

        systems = sorted(list(set(pairs_df["system1"]).union(set(pairs_df["system2"]))))
        color_map = systems_by_color.copy()

        graph_obj = nx.DiGraph()
        for _, row in pairs_df.iterrows():
            graph_obj.add_edge(
                row["Condition1"], row["Condition2"], weight=row["NPatients"]
            )

        system_nodes = {system: [] for system in systems}
        for node in graph_obj.nodes():
            for system in systems:
                if (
                    system in pairs_df[pairs_df["Condition1"] == node]["system1"].values
                    or system
                    in pairs_df[pairs_df["Condition2"] == node]["system2"].values
                ):
                    system_nodes[system].append(node)
                    break

        pos = {}
        num_nodes = len(graph_obj.nodes())
        angle_step = 2 * np.pi / num_nodes if num_nodes > 0 else 0
        current_angle = 0
        radius_scale = 1

        for system, nodes in system_nodes.items():
            for node in nodes:
                pos[node] = (
                    np.cos(current_angle) * radius_scale,
                    np.sin(current_angle) * radius_scale,
                )
                current_angle += angle_step

        label_pos = {
            node: (
                np.cos(np.arctan2(coords[1], coords[0]))
                * (radius_scale + node_label_position),
                np.sin(np.arctan2(coords[1], coords[0]))
                * (radius_scale + node_label_position),
            )
            for node, coords in pos.items()
        }

        fig, ax = plt.subplots(figsize=(10, 10))

        for system, nodes in system_nodes.items():
            if nodes:
                sizes = [node_sizes.get(node, 300) for node in nodes]
                nx.draw_networkx_nodes(
                    graph_obj,
                    pos,
                    nodelist=nodes,
                    node_color=[color_map.get(system, "#CCCCCC")],
                    node_size=sizes,
                    ax=ax,
                    alpha=0.8,
                )

        edges = graph_obj.edges(data=True)
        if edges:
            edge_weights = [d["weight"] / reduce_edge_width_by for (_, _, d) in edges]
            nx.draw_networkx_edges(
                graph_obj,
                pos,
                edgelist=edges,
                width=edge_weights,
                edge_color="gray",
                arrowstyle="-|>",
                arrowsize=10,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1",
                ax=ax,
            )

        nx.draw_networkx_labels(
            graph_obj, label_pos, font_size=9, font_family="sans-serif", ax=ax
        )

        plt.title("Disease Condition Network by Body System")
        plt.axis("off")

        if save_fig:
            plt.savefig("pair_plot.svg", dpi=100, bbox_inches="tight")

        plt.box(False)  # Remove frame

        return fig


if __name__ == "__main__":
    # Example usage
    import os

    # Load example data
    pairs_df = pd.read_csv("../example_datasets/condition_pairs.csv")
    system_df = pd.read_csv("../example_datasets/condition_systems.csv")
    freq_df = pd.read_csv("../example_datasets/condition_frequencies.csv")

    # Define system colors (example)
    system_colors = {
        "Circulatory": "#e41a1c",
        "Respiratory": "#377eb8",
        "Digestive": "#4daf4a",
        "Nervous": "#984ea3",
        "Musculoskeletal": "#ff7f00",
        "Endocrine": "#ffff33",
        "Genitourinary": "#a65628",
        "Mental": "#f781bf",
        "Other": "#999999",
    }

    # Create plot
    fig = carlota_pairs_plot(pairs_df, system_df, freq_df, system_colors)

    # Save and show the figure
    plt.tight_layout()
    if save_fig:
        plt.savefig("pair_plot.svg", dpi=100, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main_directory = os.getcwd()
    print(f"Main directory: {main_directory}")

    # Load example dataset
    patient_df = pd.read_csv(
        "../example_datasets/example_pairs_dataset.csv", delimiter=","
    )

    # Create temporal pairs object
    pairs_obj = TemporalPairs(patient_df)

    # Print some summary information
    print(f"Number of significant disease pairs: {len(pairs_obj.final_pairs_df)}")
    print(f"Number of length-3 trajectories: {len(pairs_obj.trajectories_df_length_3)}")
    print(f"Number of length-4 trajectories: {len(pairs_obj.trajectories_df_length_4)}")

    # plot pairs
    pairs_obj.pair_plot()
