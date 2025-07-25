import numpy as np

import matplotlib.pyplot as plt

from mpr_tvb_vbjax_default_parameters import default_values
from mpr_tvb_vbjax_test_functions import create_meshgrid_linspace, load_or_generate_data_for_testcase, run_tvb_implementation, run_vbjax_implementation


# Sanity check that we're looking at the correct place in the phase plane
def plot_phase_plane_and_random_points(test_case):
    args = [run_tvb_implementation(test_case), run_vbjax_implementation(test_case)]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    for id, res in enumerate(args):
        mpr_res_product = np.stack(res, axis=2)
        U = mpr_res_product[:, :, 0]
        V = mpr_res_product[:, :, 1]

        stream_plot = axs[id].streamplot(*create_meshgrid_linspace(test_case), U, V, color=np.sqrt(U**2 + V**2), cmap="viridis")
        axs[id].scatter(*load_or_generate_data_for_testcase(test_case), color="red")

        axs[id].set_xlabel("r_linspace")
        axs[id].set_ylabel("V_linspace")
        axs[id].set_title(f"Stream Plot of MPR {id}")
        fig.colorbar(stream_plot.lines, ax=axs[id], label="Magnintude")
        axs[id].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_phase_plane_and_random_points(default_values.iloc[0])
