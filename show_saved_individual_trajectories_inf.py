import numpy as np
import os
from plotting.plotting import plot_save_statistics, plot_save_fft, plot_save_plane
from bifurcation_diagrams_inf import create_pathname

def create_dir(filepath):
    plot_save_folder_path = os.path.dirname(filepath)

    # Create folder if it does not exist and we are saving the image
    if save_not_plot and (not os.path.exists(plot_save_folder_path)):
        os.makedirs(plot_save_folder_path)

def plotter(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str_att,
            normalize_weights_str_o, reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights,
            se_per_contribution, keep_context, reverse_betas, beta_to_show, gaussian_scale_str, save_non_transient,
            compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, plot_range=None):

    if save_non_transient == True:
        num_transient_steps_plot_arg = num_transient_steps
    else:
        num_transient_steps_plot_arg = 0

    folder_path = create_pathname(num_feat_patterns, tentative_semantic_embedding_size,
                                  positional_embedding_size, beta_list, num_transient_steps,
                                  max_sim_steps, context_size, normalize_weights_str_att,
                                  normalize_weights_str_o, reorder_weights, se_per_contribution,
                                  correlations_from_weights, num_segments_corrs, pe_mode, keep_context,
                                  reverse_betas, gaussian_scale_str, save_non_transient,
                                  compute_inf_normalization, scaling_o, scaling_att)
    ini_token_mode_str = ""
    if ini_token_from_w != 0:
        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
    stats_data_path = (folder_path + "/stats/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx)
                       + ini_token_mode_str + ".npz")

    # Search idx for the beta nearest to beta_to_show
    beta_to_show_idx = np.searchsorted(beta_list, beta_to_show)

    # Load data
    data = np.load(stats_data_path)
    image_format = ".jpeg"

    # Load each stat and plot/save it
    for stat_name in stats_to_save_plot:

        stat_results_beta_list = data[f"{stat_name}_results_beta_list"]
        stat_results = stat_results_beta_list[beta_to_show_idx]


        plot_save_path_traj = (folder_path + f"/indiv_traj/seed-{str(seed)}/{stat_name}/beta-{beta_to_show}-ini_token_idx-" +
                          str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

        plot_save_path_fft = (folder_path + f"/indiv_traj/seed-{str(seed)}/{stat_name}/fft-beta-{beta_to_show}-ini_token_idx-" +
                          str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

        plot_save_folder_path = os.path.dirname(plot_save_path_traj)

        # Create folder if it does not exist and we are saving the image
        if save_not_plot and (not os.path.exists(plot_save_folder_path)):
            os.makedirs(plot_save_folder_path)

        title = (
            f"MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed} "
            f"BETA={beta_to_show} NUM_TRANSIENT={num_transient_steps}")


        if plot_range is None:
            plot_save_statistics(stat_results[num_transient_steps:, :], stat_name, num_feat_patterns,
                                 max_sim_steps-num_transient_steps, show_max_num_patterns=num_feat_patterns,
                                 save_not_plot=save_not_plot, save_path=plot_save_path_traj, title=title)
        else:

            rg = range(plot_range[0], plot_range[1])
            plot_save_statistics(stat_results[rg, :], stat_name, num_feat_patterns,
                                 len(rg), min_num_step=num_transient_steps + plot_range[0],
                                 show_max_num_patterns=num_feat_patterns,
                                 save_not_plot=save_not_plot, save_path=plot_save_path_traj, title=title,
                                 plot_hilbert=False)


        plot_save_fft(stat_results[num_transient_steps_plot_arg:, :], stat_name, num_feat_patterns,
                      max_sim_steps - num_transient_steps,
                      show_max_num_patterns=num_feat_patterns,
                      save_not_plot=save_not_plot, save_path=plot_save_path_fft)


    # 3 feats
    stats_to_plot = [["mo_se", "att", "mo_se"], ["mo_se", "att", "mo_se"]]
    if num_feat_patterns == 3:
        feat_idx = [[0, 1, 2], [1, 0, 1]]
    elif num_feat_patterns == 2:
        feat_idx = [[0, 0, 0], [1, 1, 1]]
    elif num_feat_patterns == 1:
        stats_to_plot = [["mo", "mk", "mk"], ["mv", "mq", "mv"]]
        feat_idx = [[0, 0, 0], [0, 0, 0]]

    plot_save_path_plane = (folder_path + f"/indiv_traj/latex/seed-{str(seed)}/planes"
                            + f"/plane-beta-{beta_to_show}-ini_token_idx-" +
                            str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

    create_dir(plot_save_path_plane)

    stat_results_beta_list_0 = [data[f"{stats_to_plot[0][0]}_results_beta_list"][beta_to_show_idx],
                                data[f"{stats_to_plot[0][1]}_results_beta_list"][beta_to_show_idx],
                                data[f"{stats_to_plot[0][2]}_results_beta_list"][beta_to_show_idx]]
    stat_results_beta_list_1 = [data[f"{stats_to_plot[1][0]}_results_beta_list"][beta_to_show_idx],
                                data[f"{stats_to_plot[1][1]}_results_beta_list"][beta_to_show_idx],
                                data[f"{stats_to_plot[1][2]}_results_beta_list"][beta_to_show_idx]]

    plot_save_plane(stat_results_beta_list_0,
                    stat_results_beta_list_1, max_sim_steps - num_transient_steps, feat_idx,
                    tag_names=stats_to_plot, save_path=plot_save_path_plane, save_not_plot=save_not_plot)

    # # 3 feats
    # stats_to_plot = ["att", "att"]
    #
    # plot_save_path_plane = ( folder_path + f"/indiv_traj/seed-{str(seed)}/{stats_to_plot[0]}-{stats_to_plot[1]}"
    #                        + f"/plane-beta-{beta_to_show}-ini_token_idx-" +
    #                        str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)
    #
    # # Create folder if it does not exist and we are saving the image
    # if save_not_plot and (not os.path.exists(os.path.dirname(plot_save_path_plane))):
    #     os.makedirs(os.path.dirname(plot_save_path_plane))
    #
    # stat_results_beta_list_0 = data[f"{stats_to_plot[0]}_results_beta_list"][beta_to_show_idx]
    # stat_results_beta_list_1 = data[f"{stats_to_plot[1]}_results_beta_list"][beta_to_show_idx]
    # feats_reorder = np.roll(np.arange(num_feat_patterns), -1)
    # plot_save_plane(stat_results_beta_list_0,
    #                 stat_results_beta_list_1[:, feats_reorder],
    #                 num_feat_patterns, max_sim_steps - num_transient_steps, show_max_num_patterns=num_feat_patterns,
    #                 tag_names=stats_to_plot, beta=beta_to_show,
    #                 save_not_plot=save_not_plot, save_path=plot_save_path_plane)


if __name__ == "__main__":
    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 2
    context_size = 2**positional_embedding_size

    # New
    # 1 pat: seed 2 0.25, 0.27
    # 2 pat: seed 18 0.8, 1.3
    # 3 pat: seed 1 0.35, 0.8

    # Create variables for the Hopfield Transformer (HT)
    seed = 1
    beta_list = np.linspace(0, 4, 1500)
    # se_per_contribution = tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)
    se_per_contribution = 0.98
    num_feat_patterns = 3
    num_transient_steps = 100000
    saved_steps = 20000
    max_sim_steps = num_transient_steps + saved_steps
    keep_context = True
    reverse_betas = False
    beta_to_show = 0.3729   # We'll find the nearest beta in the defined range

    plot_window = 1000
    offset = 10000 + plot_window
    plot_range = [saved_steps - offset - 1, saved_steps - offset + plot_window -1]    # Index of the range of steps want to plot within the trajectory


    ini_token_idx = 0
    ini_token_from_w = 1
    reorder_weights = False
    normalize_weights_str_o = "N"
    normalize_weights_str_att = "N**2*np.sqrt(M)"
    scaling_o = 1
    scaling_att = 100
    compute_inf_normalization = True
    correlations_from_weights = 3  # 0 use gaussian corrs, 1 create from weight matrices, 2 uniform means, 3 segments
    num_segments_corrs = 3         # Only applicable if correlations_from_weights=3
    pe_mode = 0
    gaussian_scale = "0.50"         # Only applicable if correlations_from_weights=0
    save_not_transient = False
    save_not_plot = False

    if context_size > 2**positional_embedding_size:
        raise("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise("You cannot discard more timesteps than you are simulating.")

    # stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    stats_to_save_plot = ["mo_se"]


    plotter(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
           num_transient_steps, max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str_att,
           normalize_weights_str_o, reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights,
           se_per_contribution, keep_context, reverse_betas, beta_to_show, gaussian_scale, save_not_transient,
           compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, plot_range)
