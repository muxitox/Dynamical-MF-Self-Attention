import numpy as np
import os
from simulate_mf_trajectory import plot_save_statistics

def plotter(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str, reorder_weights, save_not_plot,
            stats_to_save_plot, correlations_from_weights, se_per_contribution, keep_context, beta_to_show):


    # Create folder path name as defined in the runner function
    folder_path = ("results/infN-correlations_from_weights-" + str(int(correlations_from_weights)) +
                   "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-" +
                   str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                   + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-" +
                   normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                   + "/max_sim_steps-" + str(max_sim_steps) + "-context_size-" + str(context_size) +
                   "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1]) +
                   "-num_betas-" + str(len(beta_list)) + "-keep_context-" + str(int(keep_context)))

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


        plot_save_path = (folder_path + f"/indiv_traj/seed-{str(seed)}/{stat_name}/beta-{beta_to_show}-ini_token_idx-" +
                          str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + ".png")


        plot_save_folder_path = os.path.dirname(plot_save_path)

        # Create folder if it does not exist and we are saving the image
        if save_not_plot and (not os.path.exists(plot_save_folder_path)):
            os.makedirs(plot_save_folder_path)

        plot_save_statistics(stat_results[num_transient_steps:,:], stat_name, num_feat_patterns, max_sim_steps-num_transient_steps,
                             show_max_num_patterns=None,
                             save_not_plot=save_not_plot, save_path=plot_save_path)



if __name__ == "__main__":
    # Instantiate vocabulary
    tentative_semantic_embedding_size = 100
    positional_embedding_size = 4
    context_size = 2**positional_embedding_size

    # Create variables for the Hopfield Transformer (HT)
    seed = 5
    beta_list = np.linspace(0, 4, 1500)
    se_per_contribution = 0.95
    num_feat_patterns = 3
    num_transient_steps = 1024  # 0 if we want to show the trajectory since the beginning
    max_sim_steps = 1536
    keep_context = False
    beta_to_show = 2    # We'll find the nearest beta in the defined range


    if context_size > 2**positional_embedding_size:
        raise("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise("You cannot discard more timesteps than you are simulating.")

    ini_token_idx = 2
    reorder_weights = False
    normalize_weights_str = "np.sqrt(N*M)"
    correlations_from_weights = False
    save_not_plot = True

    stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    # stats_to_save_plot = ["mo", "mo_se"]


    plotter(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str, reorder_weights, save_not_plot,
            stats_to_save_plot, correlations_from_weights, se_per_contribution, keep_context, beta_to_show)
