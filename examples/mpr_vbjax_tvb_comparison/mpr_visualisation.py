from matplotlib import pyplot as plt
import seaborn as sns

from mpr_config import MPRConfig
from tvb_mpr_model import TvbMPRModel
from vbjax_model import VBJaxModel


def graph_noise_comparison(initial_conditons_seed, number_of_tests):
    # Set nice Seaborn style
    sns.set_theme(style="whitegrid", context="talk", palette="muted")

    vbjax_noise_graph = []
    tvb_noise_graph = []
    config = MPRConfig(initial_conditions_seed=initial_conditons_seed)
    config.init_cond_for_noise()

    # Run model without noise
    config.noise_seed = 0
    config.noise = 1.0
    tvb_noise_graph = TvbMPRModel(config).run()[0][1].flatten()

    # Run model with noise
    config.noise_seed = 0 + number_of_tests
    vbjax_noise_graph = VBJaxModel(config).run()[1].flatten()

    # Plot
    plt.figure(figsize=(10, 6))

    # Use smooth density instead of just bars (still semi-transparent)
    sns.histplot(
        tvb_noise_graph,
        bins=50,
        color="#e74c3c",
        kde=True,
        label="tvb",
        stat="density",
        alpha=0.5,
    )
    sns.histplot(
        vbjax_noise_graph,
        bins=50,
        color="#224cac",
        kde=True,
        label="vbjax",
        stat="density",
        alpha=0.5,
    )

    # Add nice labels and legend
    plt.title("Comparison of Model Output Distributions\nWith vs. Without Noise", fontsize=18, pad=15)
    plt.xlabel("Model Output", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Condition", fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    graph_noise_comparison(15, 1000)
