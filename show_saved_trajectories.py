import numpy as np
import os
from plotting.plotting import plot_save_statistics, plot_save_fft, plot_save_plane
from bifurcation_diagrams import create_pathname
from utils import create_dir_from_filepath
import yaml

def plotter(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list, cfg,
            stats_to_save_plot, beta_to_show, load_from_context_mode=0, plot_range=None):

    if cfg["save_non_transient"] == True:
        num_transient_steps_plot_arg = cfg["num_transient_steps"]
    else:
        num_transient_steps_plot_arg = 0

    folder_path = create_pathname(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                                  load_from_context_mode, cfg)



    # Search idx for the beta nearest to beta_to_show
    beta_to_show_idx = np.searchsorted(worker_values_list, beta_to_show)
    print("beta", beta_to_show_idx, worker_values_list[beta_to_show_idx])

    ini_token_mode_str = ""
    ini_token_from_w = cfg["ini_token_from_w"]
    if ini_token_from_w != 0:
        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
    stats_data_path = (folder_path + "/stats/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx)
                       + ini_token_mode_str + "-beta_idx-" + str(beta_to_show_idx) + ".npz")


    # Load data
    data = np.load(stats_data_path)
    image_format = ".jpeg"

    # Load each stat and plot/save it
    for stat_name in stats_to_save_plot:

        stat_results_beta_list = data[f"{stat_name}_results_beta"]
        stat_results = stat_results_beta_list

        plot_save_path_traj = (folder_path + f"/indiv_traj/seed-{str(seed)}/{stat_name}/beta-{beta_to_show}-ini_token_idx-" +
                          str(ini_token_idx) + "-transient_steps-" + str(cfg["num_transient_steps"]) + image_format)

        plot_save_path_fft = (folder_path + f"/indiv_traj/seed-{str(seed)}/{stat_name}/fft-beta-{beta_to_show}-ini_token_idx-" +
                          str(ini_token_idx) + "-transient_steps-" + str(cfg["num_transient_steps"]) + image_format)

        plot_save_folder_path = os.path.dirname(plot_save_path_traj)

        # Create folder if it does not exist and we are saving the image
        if cfg["save_not_plot"] and (not os.path.exists(plot_save_folder_path)):
            os.makedirs(plot_save_folder_path)

        # title = (
        #     f"MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed} "
        #     f"BETA={beta_to_show} NUM_TRANSIENT={num_transient_steps}")

        title = fr"$\beta$ = {beta_to_show}"

        if plot_range is None:
            plot_range = [0, None]

        rg = range(plot_range[0], plot_range[1])
        show_1_feat = 1
        plot_save_statistics(stat_results[rg, :], stat_name, num_feat_patterns,
                             len(rg), min_num_step=cfg["num_transient_steps"] + plot_range[0],
                             show_max_num_patterns=num_feat_patterns,
                             save_not_plot=cfg["save_not_plot"], save_path=plot_save_path_traj, title=title,
                             plot_hilbert=False, show_1_feat=show_1_feat)


        plot_save_fft(stat_results[num_transient_steps_plot_arg:, :], stat_name, num_feat_patterns,
                      cfg["max_sim_steps"] - cfg["num_transient_steps"],
                      show_max_num_patterns=num_feat_patterns, save_not_plot=cfg["save_not_plot"],
                      save_path=plot_save_path_fft, title=title, show_1_feat=show_1_feat)


    # 3 feats
    stats_to_plot = [["mo_se"], ["mo_se"]]
    feat_idx = [[0], [1]]

    plot_save_path_plane = (folder_path + f"/indiv_traj/latex/seed-{str(seed)}/planes"
                            + f"/plane-beta-{beta_to_show}-ini_token_idx-" +
                            str(ini_token_idx) + "-transient_steps-" + str(cfg["num_transient_steps"]) + image_format)

    create_dir_from_filepath(plot_save_path_plane)

    stat_results_beta_list_0 = [data[f"{stats_to_plot[0][0]}_results_beta"]]
    stat_results_beta_list_1 = [data[f"{stats_to_plot[1][0]}_results_beta"]]

    plot_save_plane(stat_results_beta_list_0,
                    stat_results_beta_list_1, cfg["max_sim_steps"] - cfg["num_transient_steps"], feat_idx,
                    tag_names=stats_to_plot, save_path=plot_save_path_plane, save_not_plot=cfg["save_not_plot"],
                    title=title)


if __name__ == "__main__":
    # Load cfg
    cfg_path = 'cfgs/bif_diagram_inf_0_zoom-in.yaml'
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    num_bifurcation_values = 4001  # Number of x values to examine in the bifurcation diagram

    worker_values_list = np.linspace(cfg["min_bifurcation_value"], cfg["max_bifurcation_value"],
                                     num_bifurcation_values)  # Betas or Epsilon values

    seed = 1  # List of seeds to review
    num_feat_patterns = 3  # List of number of features for which to initialize the model
    ini_token_idx = 0
    load_from_last_chpt = True
    beta_to_show = 0.5

    plot_window = 250
    offset = 10000 + plot_window
    plot_range = [offset, offset + plot_window]    # Index of the range of steps want to plot within the trajectory


    if context_size > 2**positional_embedding_size:
        raise Exception("The positional embedding cannot cover the whole context size.")
    if cfg["num_transient_steps"] > cfg["max_sim_steps"]:
        raise Exception("You cannot discard more timesteps than you are simulating.")
    if plot_range[1] >=  cfg["max_sim_steps"] - cfg["num_transient_steps"]:
        raise Exception("You are trying to plot more steps than simulated. Redefine plot_range.")

    # stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    stats_to_save_plot = ["mo_se"]

    if load_from_last_chpt:
        load_from_context_mode = 1
    else:
        load_from_context_mode = 0

    plotter(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list, cfg,
            stats_to_save_plot, beta_to_show, load_from_context_mode=load_from_context_mode, plot_range=plot_range)
