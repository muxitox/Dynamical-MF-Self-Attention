import numpy as np
import os
from plotting.plotting import plot_save_statistics, plot_save_fft, plot_save_plane

def plotter(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str,
            reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights, se_per_contribution,
            keep_context, reverse_betas, beta_to_show, gaussian_scale_str, save_non_transient, plot_range=None):

    reverse_betas_str = ""
    if reverse_betas:
        reverse_betas_str = "-reverse_betas"

    if correlations_from_weights != 0:
        gaussian_scale_name_str = ""
    else:
        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

    if save_non_transient == True:
        save_non_transient_str = ""
        num_transient_steps_plot_arg = num_transient_steps
    else:
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"
        num_transient_steps_plot_arg = 0

    folder_path = ("results/infN-correlations_from_weights-" + str(correlations_from_weights)
                   + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                   + str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                   + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-"
                   + normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                   + "-num_segments_corrs-" + str(num_segments_corrs) + "-pe_mode-" + str(pe_mode)
                   + gaussian_scale_name_str + "/max_sim_steps-" + str(max_sim_steps)
                   + save_non_transient_str + "-context_size-" + str(context_size)
                   + "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1])
                   + "-num_betas-" + str(len(beta_list)) + f"{reverse_betas_str}-keep_context-"
                   + str(int(keep_context)))

    stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) +
                       ".npz")

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
                                 len(rg), min_num_step=max_sim_steps+plot_range[0],
                                 show_max_num_patterns=num_feat_patterns,
                                 save_not_plot=save_not_plot, save_path=plot_save_path_traj, title=title)

        plot_save_fft(stat_results[num_transient_steps_plot_arg:, :], stat_name, num_feat_patterns,
                      max_sim_steps - num_transient_steps,
                      show_max_num_patterns=num_feat_patterns,
                      save_not_plot=save_not_plot, save_path=plot_save_path_fft)

    # 3 feats

    stats_to_plot = ["att", "att"]

    plot_save_path_plane = ( folder_path + f"/indiv_traj/seed-{str(seed)}/{stats_to_plot[0]}-{stats_to_plot[1]}"
                           + f"/plane-beta-{beta_to_show}-ini_token_idx-" +
                           str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

    # Create folder if it does not exist and we are saving the image
    if save_not_plot and (not os.path.exists(os.path.dirname(plot_save_path_plane))):
        os.makedirs(os.path.dirname(plot_save_path_plane))

    stat_results_beta_list_0 = data[f"{stats_to_plot[0]}_results_beta_list"][beta_to_show_idx]
    stat_results_beta_list_1 = data[f"{stats_to_plot[1]}_results_beta_list"][beta_to_show_idx]
    feats_reorder = np.roll(np.arange(num_feat_patterns), -1)
    plot_save_plane(stat_results_beta_list_0,
                    stat_results_beta_list_1[:, feats_reorder],
                    num_feat_patterns, max_sim_steps - num_transient_steps, show_max_num_patterns=num_feat_patterns,
                    tag_names=stats_to_plot, beta=beta_to_show,
                    save_not_plot=save_not_plot, save_path=plot_save_path_plane)


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
    beta_list = np.linspace(0.35, 0.8, 1000)
    se_per_contribution = tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)
    num_feat_patterns = 3
    num_transient_steps = 100000
    saved_steps = 20000
    max_sim_steps = num_transient_steps + saved_steps
    keep_context = False
    reverse_betas = False
    beta_to_show = 0.3729   # We'll find the nearest beta in the defined range

    plot_window = 5000
    offset = 15000 + plot_window
    plot_range = [saved_steps - offset - 1, saved_steps - offset + plot_window -1]    # Index of the range of steps want to plot within the trajectory

    ini_token_idx = 0
    reorder_weights = False
    normalize_weights_str = "np.sqrt(N)*M"
    correlations_from_weights = 3  # 0 use gaussian corrs, 1 create from weight matrices, 2 uniform means, 3 segments
    num_segments_corrs = 3         # Only applicable if correlations_from_weights=3
    pe_mode = 0
    gaussian_scale = "0.50"         # Only applicable if correlations_from_weights=0
    save_not_transient = False
    save_not_plot = True

    if context_size > 2**positional_embedding_size:
        raise("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise("You cannot discard more timesteps than you are simulating.")

    # stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    stats_to_save_plot = ["mo_se"]

    plotter(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str,
            reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights, se_per_contribution,
            keep_context, reverse_betas, beta_to_show, gaussian_scale, save_not_transient, plot_range)
