import numpy as np
from plotting.plotting import plot_bifurcation_diagram
import os

def plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str, reorder_weights, save_not_plot,
            stats_to_save_plot, correlations_from_weights, se_per_contribution, keep_context, reverse_betas,
            min_beta_to_show, max_beta_to_show, gaussian_scale_str):


    reverse_betas_str = ""
    if reverse_betas:
        reverse_betas_str = "-reverse_betas"

    if correlations_from_weights != 0:
        gaussian_scale_name_str = ""
    else:
        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:
            for ini_token_idx in ini_tokens_list:

                folder_path = ("results/infN-correlations_from_weights-" + str(correlations_from_weights)
                               + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                               + str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                               + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-"
                               + normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                               + "-num_segments_corrs-" + str(num_segments_corrs) + "-pe_mode-" + str(pe_mode)
                               + gaussian_scale_name_str + "/max_sim_steps-" + str(max_sim_steps)
                               + "-context_size-" + str(context_size)
                               + "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1])
                               + "-num_betas-" + str(len(beta_list)) + f"{reverse_betas_str}-keep_context-"
                               + str(int(keep_context)))

                stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) +
                                   ".npz")

                # Load data
                data = np.load(stats_data_path)

                show_max_num_patterns = 6

                min_beta_idx = np.searchsorted(beta_list, min_beta_to_show)
                max_beta_idx = np.searchsorted(beta_list, max_beta_to_show) + 1

                # Load each stat and plot/save it
                for stat_name in stats_to_save_plot:

                    stat_results_beta_list = data[f"{stat_name}_results_beta_list"]

                    # Create folder if it does not exist and we are saving the image
                    if save_not_plot and (not os.path.exists(folder_path + f"/{stat_name}/")):
                        os.makedirs(folder_path + f"/{stat_name}/")

                    image_format = ".jpeg"

                    fig_save_path = (folder_path + f"/{stat_name}/seed-" + str(seed) + "-ini_token_idx-" +
                                      str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) +
                                      f"-min_beta-{min_beta_to_show}-max_beta-{max_beta_to_show}{image_format}")

                    plot_bifurcation_diagram(stat_results_beta_list[min_beta_idx:max_beta_idx], beta_list[min_beta_idx:max_beta_idx],
                                             num_feat_patterns, fig_save_path,
                                             num_transient_steps, feat_name=stat_name,
                                             show_max_num_patterns=show_max_num_patterns, save_not_plot=save_not_plot)


if __name__ == "__main__":

    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 3
    context_size = 2 ** positional_embedding_size

    # VARS FOR LOADING CHECKPOINTS
    # Create variables for the Hopfield Transformer (HT)
    seed_list = [4]
    beta_list = np.linspace(0, 3, 1000)
    se_per_contribution = tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)
    print(se_per_contribution)
    num_feat_patterns_list = [3]
    ini_tokens_list = [0]
    num_transient_steps = 2560
    max_sim_steps = 4068
    keep_context = True
    reverse_betas = False

    # Specific stats for plotting
    min_beta_to_show = 0
    max_beta_to_show = 3

    # seed_list = [ 1]
    # num_feat_patterns_list = [3]

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise ("You cannot discard more timesteps than you are simulating.")

    reorder_weights = False
    normalize_weights_str = "np.sqrt(N)*M"
    correlations_from_weights = 3  # 0 use gaussian corrs, 1 create from weight matrices, 2 uniform means, 3 segments
    num_segments_corrs = 3         # Only applicable if correlations_from_weights=3
    pe_mode = 0
    gaussian_scale = "0.5"         # Only applicable if correlations_from_weights=0
    save_not_plot = False

    # stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    stats_to_save_plot = ["mo", "mo_se"]

    plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str,
            reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights, se_per_contribution,
            keep_context, reverse_betas, min_beta_to_show, max_beta_to_show, gaussian_scale)

