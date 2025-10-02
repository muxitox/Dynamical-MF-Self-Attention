import numpy as np
import os
from plotting.plotting import plot_save_statistics, plot_save_fft, plot_save_plane, plot_lyapunov_graphs
import matplotlib.pyplot as plt
from utils import create_dir_from_filepath
import yaml

def plotter(worker_x_values_list , x_cutting_points_idxs, cfg, exp_dir, save_not_plot=False):

    if cfg["save_non_transient"] == True:
        num_transient_steps_plot_arg = cfg["num_transient_steps"]
    else:
        num_transient_steps_plot_arg = 0

    save_basepath = (exp_dir + f"/phase_portrait_breakout/")

    if save_not_plot and (not os.path.exists(save_basepath)):
        os.makedirs(save_basepath)



    for x_idx in x_cutting_points_idxs:

        stats_data_path = (exp_dir + "/stats/beta_idx-" + str(x_idx) + ".npz")

        # Load data
        data = np.load(stats_data_path)
        image_format = ".jpeg"

        lyapunov_0 = data["S"][0]

        #################
        ### Plot planes
        #################

        plot_save_path_plane = (save_basepath + f"/planes/"
                                + f"/plane-beta-{x_idx}" + "-transient_steps-" +
                                str(cfg["num_transient_steps"]) + image_format)

        if save_not_plot:
            create_dir_from_filepath(plot_save_path_plane)

        nrows = 2
        ncols = 3
        fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8), constrained_layout=True)

        feat_suffix = "_results_beta"
        # Define the statistics you want to plot against each other
        # In this case the feature mo with only the semantic information
        stats_to_plot = [["mo_se", "mo_se"], ["mo_se", "mo_se"], ["mo_se", "mo_se"],
                         ["att", "att"], ["att", "att"], ["att", "att"]]
        # Define the index of the features you want to compare against each other
        feat_idx = [[0, 1], [0, 2], [1, 2], [0, 1], [0, 2], [1, 2]]

        flat_ax = ax.ravel()

        for plot_i in range(len(flat_ax)):
            # Load needed statistics
            stat_results_beta_0 = data[stats_to_plot[plot_i][0] + feat_suffix][:, feat_idx[plot_i][0]]
            stat_results_beta_1 = data[stats_to_plot[plot_i][1] + feat_suffix][:, feat_idx[plot_i][1]]

            plot_save_plane(stat_results_beta_0,
                            stat_results_beta_1, cfg["max_sim_steps"] - cfg["num_transient_steps"], feat_idx[plot_i],
                            flat_ax[plot_i], tag_names=stats_to_plot[plot_i])

        title = rf"$\beta$={worker_x_values_list[x_idx]} $\lambda_0$={lyapunov_0:0.5g}"

        fig.suptitle(title)

        if save_not_plot:
            fig.savefig(plot_save_path_plane, bbox_inches='tight')
        else:
            fig.show()

        plt.close(fig)





if __name__ == "__main__":

    exp_dir = "results_continuation/20250925_171913_zoom2_middle2"


    cfg_path = exp_dir + "/cfg.yaml"
    # Load cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    stats_to_save_plot = ["mo_se", "att"]
    save_not_plot = False     # True -> save, False -> plot

    # Create x values for the bifurcation diagram.
    worker_values_list = np.linspace(cfg["min_bifurcation_value"], cfg["max_bifurcation_value"],
                                     cfg["num_bifurcation_values"])  # Betas or Epsilon values

    x_cutting_points_idxs = [18, 47, 50, 95, 118, 166, 235, 300, 363, 375, 382, 390, 455, 485]

    plotter(worker_values_list, x_cutting_points_idxs, cfg, exp_dir, save_not_plot=save_not_plot)
