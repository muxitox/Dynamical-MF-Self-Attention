import numpy as np
import os
from plotting.plotting import plot_save_statistics, plot_save_fft

def plotter(num_feat_patterns, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str, reorder_weights, save_not_plot,
            stats_to_save_plot, keep_context, reverse_betas, beta_to_show):

    reverse_betas_str = ""
    if reverse_betas:
        reverse_betas_str = "-reverse_betas"

    # Create folder path name as defined in the runner function
    folder_path = ("results/" + "se_size-" + str(semantic_embedding_size) + "-pe_size-" +
                   str(positional_embedding_size)
                   + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-" +
                   normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                   + "/max_sim_steps-" + str(max_sim_steps) + "-context_size-" + str(context_size) +
                   "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1]) +
                   "-num_betas-" + str(len(beta_list)) + f"{reverse_betas_str}-keep_context-" + str(int(keep_context)))

    stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) +
                       ".npz")

    # Load data
    data = np.load(stats_data_path)

    show_max_num_patterns = 6

    # Load each stat and plot/save it
    for stat_name in stats_to_save_plot:

        stat_results_beta_list = data[f"{stat_name}_results_beta_list"]

        # Search idx for the beta nearest to beta_to_show
        beta_to_show_idx = np.searchsorted(beta_list, beta_to_show)

        stat_results = stat_results_beta_list[beta_to_show_idx]

        image_format = ".jpeg"

        plot_save_path_traj = (folder_path + f"/indiv_traj/seed-{str(seed)}/{stat_name}/beta-{beta_to_show}-ini_token_idx-" +
                          str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

        plot_save_path_fft = (
                    folder_path + f"/indiv_traj/seed-{str(seed)}/{stat_name}/fft-beta-{beta_to_show}-ini_token_idx-" +
                    str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

        plot_save_folder_path = os.path.dirname(plot_save_path_traj)

        # Create folder if it does not exist and we are saving the image
        if save_not_plot and (not os.path.exists(plot_save_folder_path)):
            os.makedirs(plot_save_folder_path)

        plot_save_statistics(stat_results[num_transient_steps:,:], stat_name, num_feat_patterns, max_sim_steps-num_transient_steps,
                             show_max_num_patterns=show_max_num_patterns,
                             save_not_plot=save_not_plot, save_path=plot_save_path_traj)

        plot_save_fft(stat_results[num_transient_steps:, :], stat_name, num_feat_patterns,
                      max_sim_steps - num_transient_steps,
                      show_max_num_patterns=show_max_num_patterns,
                      save_not_plot=save_not_plot, save_path=plot_save_path_fft)




if __name__ == "__main__":
    # Instantiate vocabulary
    semantic_embedding_size = 100
    positional_embedding_size = 4
    context_size = 2**positional_embedding_size

    # Create variables for the Hopfield Transformer (HT)
    seed = 3
    beta_list = np.linspace(0, 4, 1500)
    num_feat_patterns = 16
    num_transient_steps = 1024  # 0 if we want to show the trajectory since the beginning
    max_sim_steps = 1536
    keep_context = True
    reverse_betas = False
    beta_to_show = 1.8    # We'll find the nearest beta in the defined range

    ini_token_idx = 0
    reorder_weights = False
    normalize_weights_str = "np.sqrt(N*M)"
    correlations_from_weights = False
    save_not_plot = True

    if context_size > 2**positional_embedding_size:
        raise("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise("You cannot discard more timesteps than you are simulating.")

    # stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    stats_to_save_plot = ["mo_se"]

    plotter(num_feat_patterns, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str, reorder_weights, save_not_plot,
            stats_to_save_plot, keep_context, reverse_betas, beta_to_show)