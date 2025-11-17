from matplotlib import pyplot as plt
import seaborn as sns

from config import Config
from tvb_model import TvbModel


def graph_noise_comparison(initial_conditons_seed, number_of_tests):
    # Set nice Seaborn style
    sns.set_theme(style="whitegrid", context="talk", palette="muted")

    results_low_noise = []
    results_with_noise = []
    config = Config(initial_conditions_seed=initial_conditons_seed)
    config.init_cond_for_noise()

    # Run model without noise
    config.noise_seed = 0
    config.noise = 10.0
    results_with_noise = TvbModel(config).run().flatten()

    # Run model with noise
    config.noise_seed = 0 + number_of_tests
    config.noise = 0.1
    results_low_noise = TvbModel(config).run().flatten()

    # Plot
    plt.figure(figsize=(10, 6))

    # Use smooth density instead of just bars (still semi-transparent)
    sns.histplot(
        results_with_noise,
        bins=50,
        color="#e74c3c",
        kde=True,
        label="High Noise",
        stat="density",
        alpha=0.5,
    )
    sns.histplot(
        results_low_noise,
        bins=50,
        color="#224cac",
        kde=True,
        label="Low Noise",
        stat="density",
        alpha=0.5,
    )

    # Add nice labels and legend
    plt.title(
        "Comparison of Model Output Distributions\nWith vs. Without Noise", fontsize=18, pad=15
    )
    plt.xlabel("Model Output", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Condition", fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    graph_noise_comparison(15, 1000)
