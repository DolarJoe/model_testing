import os

import numpy as np
import tvb_data.connectivity as conn_data
from matplotlib import pyplot as plt
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.lab import connectivity, plot_connectivity


def generate_matrices(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates a random connectivity matrix for testing."""
    connectivity_matrix = np.random.rand(n_nodes, n_nodes)
    connectivity_matrix /= np.sum(connectivity_matrix)
    np.fill_diagonal(connectivity_matrix, 0)
    return connectivity_matrix, np.zeros((n_nodes, n_nodes))  # Delay matrix not used in random case


def load_matrices_from_tvb() -> tuple[np.ndarray, np.ndarray]:
    """Loads the connectivity and delay matrices from TVB's built-in datasets."""
    data_path = os.path.join(os.path.dirname(conn_data.__file__), "connectivity_96.zip")
    white_matter = Connectivity.from_file(data_path)
    white_matter.configure()

    connectivity_matrix = white_matter.weights
    connectivity_matrix /= np.max(connectivity_matrix)
    delay_matrix = white_matter.tract_lengths

    return connectivity_matrix, delay_matrix


def show_connectivity(connectivity_matrix: np.ndarray, delay_matrix: np.ndarray) -> None:
    """Visualizes the connectivity matrix using TVB's plotting functions."""
    conn = connectivity.Connectivity()
    n_regions = connectivity_matrix.shape[0]

    conn.tract_lengths = delay_matrix
    conn.weights = connectivity_matrix
    conn.centres = np.zeros((n_regions, 3))
    conn.region_labels = np.array([f"{i}" for i in range(n_regions)])
    conn.configure()

    plot_connectivity(conn, num="tract_mode", plot_tracts=False)
    plt.show()
