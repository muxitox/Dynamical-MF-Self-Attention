import numpy as np
import os
from plotting.plotting import plot_save_statistics, plot_save_fft, plot_save_plane, plot_lyapunov_graphs
import matplotlib.pyplot as plt
from utils import create_dir_from_filepath
import yaml

def plotter(worker_values_list, beta_to_show_idx, cfg, exp_dir, plot_range = None, save_not_plot=False):

    if cfg["save_non_transient"] == True:
        num_transient_steps_plot_arg = cfg["num_transient_steps"]
    else:
        num_transient_steps_plot_arg = 0

    save_basepath = (exp_dir + f"/saved_trajectories/")

    if save_not_plot and (not os.path.exists(save_basepath)):
        os.makedirs(save_basepath)


    stats_data_path = (exp_dir + "/stats/beta_idx-" + str(beta_to_show_idx) + ".npz")


    # Load data
    data = np.load(stats_data_path)
    image_format = ".jpeg"

    # Load each stat and plot/save it
    for stat_name in stats_to_save_plot:

        stat_results_beta_list = data[f"{stat_name}_results_beta"]
        stat_results = stat_results_beta_list

        plot_save_path_traj = (save_basepath + f"/traj-beta-{beta_to_show}-transient_steps-" + str(cfg["num_transient_steps"]) + image_format)

        plot_save_path_fft = (save_basepath + f"/fft-beta-{beta_to_show}-transient_steps-" + str(cfg["num_transient_steps"]) + image_format)

        plot_save_folder_path = os.path.dirname(plot_save_path_traj)

        # Create folder if it does not exist and we are saving the image
        if save_not_plot and (not os.path.exists(plot_save_folder_path)):
            os.makedirs(plot_save_folder_path)

        # title = (
        #     f"MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed} "
        #     f"BETA={beta_to_show} NUM_TRANSIENT={num_transient_steps}")

        title = fr"$\beta$ = {worker_values_list[beta_to_show_idx]}"

        if plot_range is None:
            plot_range = [0, None]

        rg = range(plot_range[0], plot_range[1])
        show_1_feat = 1
        plot_save_statistics(stat_results[rg, :], stat_name, cfg["num_feat_patterns"],
                             len(rg), min_num_step=cfg["num_transient_steps"] + plot_range[0],
                             show_max_num_patterns=cfg["num_feat_patterns"],
                             save_not_plot=save_not_plot, save_path=plot_save_path_traj, title=title,
                             plot_hilbert=False, show_1_feat=show_1_feat)


        plot_save_fft(stat_results[num_transient_steps_plot_arg:, :], stat_name, cfg["num_feat_patterns"],
                      cfg["max_sim_steps"] - cfg["num_transient_steps"],
                      show_max_num_patterns=cfg["num_feat_patterns"], save_not_plot=save_not_plot,
                      save_path=plot_save_path_fft, title=title, show_1_feat=show_1_feat)


    #################
    ### Plot planes
    #################
    plot_save_path_plane = (save_basepath + f"/planes/"
                            + f"/plane-beta-{worker_values_list[beta_to_show_idx]}" + "-transient_steps-" +
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

    if save_not_plot:
        fig.savefig(plot_save_path_plane, bbox_inches='tight')
    else:
        fig.show()

    plt.close(fig)


    # Plot Lyapunov
    print("Lyapunov exponentsm (last 0 exponents correspond to positional encodings, we filter them when analyzing the data)")
    print(data["S"])
    print("Lyapunov exponents going to inf")
    print(data["S_inf_flag"])


    show_lyapunov_traces = True
    lya_trace_folder = (exp_dir + "/lyapunov_traces/")
    print(lya_trace_folder)
    if show_lyapunov_traces and os.path.exists(lya_trace_folder):
        lyapunov_trace = np.load(f"{lya_trace_folder}trace{beta_to_show_idx}.npz")
        S_i = lyapunov_trace["S_i"]
        S_i_mean = np.mean(S_i, axis=0)
        std = np.sqrt(np.mean((S_i - S_i_mean)**2, axis=0)/(S_i_mean.shape[0]-1))
        std_error = std/np.sqrt(S_i_mean.shape[0])

        T = np.arange(1, S_i.shape[0] + 1)

        S_i_sum = np.cumsum(S_i, axis=0) / T[:, None]

        num_last_steps = 5000
        conv_error = np.mean(np.abs(np.diff(S_i[-num_last_steps:,:], axis=0)), axis=0)
        conv_error_sum = np.mean(np.abs(np.diff(S_i_sum[-num_last_steps:,:], axis=0)), axis=0)


        print("Mean", S_i_mean[0], "Std", std[0], "StdError", std_error[0], "Converg Error", conv_error[0], "Converg Error Sum", conv_error_sum[0])


        plt.plot(S_i[-num_last_steps:,0])
        plt.ylabel(r"$\lambda_0$")
        plt.show()
        plt.close()

        plt.plot(S_i_sum[-num_last_steps:,0])
        plt.ylabel(r"$\lambda_0$")
        plt.show()
        plt.close()




if __name__ == "__main__":

    exp_dir = "results_parallel_v3/results_small_sample_zoom2"
    exp_dir = "results_continuation/20250701_150546_original"


    cfg_path = exp_dir + "/cfg.yaml"
    # Load cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    stats_to_save_plot = ["mo_se", "att"]
    save_not_plot = False     # True -> save, False -> plot

    # Define plotting range
    plot_window = 250
    offset = 10000 + plot_window
    plot_range = [offset, offset + plot_window]  # Index of the range of steps want to plot within the trajectory

    # Create x values for the bifurcation diagram.
    worker_values_list = np.linspace(cfg["min_bifurcation_value"], cfg["max_bifurcation_value"],
                                     cfg["num_bifurcation_values"])  # Betas or Epsilon values

    # It will look for the most similar beta in the experiments
    beta_to_show = 1.26
    # beta_to_show = 1.27387
    # Search idx for the beta nearest to beta_to_show
    beta_to_show_idx = np.searchsorted(worker_values_list, beta_to_show)
    # 679 chaos, 680 periodic, 681 periodic, 682 chaotic, 683 periodic
    # beta_to_show_idx = 214  # You can also define the index instead of searching
    print("beta", beta_to_show_idx, worker_values_list[beta_to_show_idx])


    plotter(worker_values_list, beta_to_show_idx, cfg, exp_dir, plot_range=plot_range, save_not_plot=save_not_plot)
