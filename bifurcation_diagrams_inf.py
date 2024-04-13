import numpy as np
from models.HopfieldTransformerPE import Embedding
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from plotting.plotting import plot_bifurcation_diagram
import os
import time


def runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
           max_sim_steps,
           context_size, num_ini_tokens, seed_list, normalize_weights_str, reorder_weights, stats_to_save_plot,
           se_per_contribution, correlations_from_weights, num_segments_corrs, pe_mode, keep_context, reverse_betas,
           gaussian_scale_str, save_non_transient):
    vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size)

    np.random.seed(0)
    ini_tokens_list = np.random.randint(2, size=(
    num_ini_tokens, tentative_semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    if reverse_betas:  # Flip vector to avoid falling in the same fixed points due to the mean-field 0 a low beta
        beta_list = np.flip(beta_list)

    min_saved_step = 0
    if not save_non_transient:
        min_saved_step = num_transient_steps

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:

            np.random.seed(seed)

            # Initialize transformer weights and create variables for storing results
            HT = HopfieldTransformerInfN(0, 0, num_feat_patterns=num_feat_patterns,
                                         positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                         context_size=context_size, max_sim_steps=max_sim_steps,
                                         min_saved_step=min_saved_step,
                                         normalize_weights_str=normalize_weights_str,
                                         reorder_weights=reorder_weights,
                                         correlations_from_weights=correlations_from_weights,
                                         num_segments_corrs=num_segments_corrs, pe_mode=pe_mode,
                                         semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                         se_per_contribution=se_per_contribution)

            for ini_token_idx in range(0, num_ini_tokens):

                # Initialize structure for saving the results for each beta
                results_beta_list = {}
                for stat_name in HT.statistics_names:
                    results_beta_list[stat_name] = []

                for beta_idx in range(0, len(beta_list)):
                    beta_o = beta_list[beta_idx]
                    beta_att = beta_list[beta_idx]
                    HT.set_betas(beta_o, beta_att)

                    if beta_idx == 0 or not keep_context:
                        # For the first beta in the series reset everything and start from scratch
                        # Reset the matrix for storing results
                        HT.reset_data()
                        # Encode initial token with position 0
                        x0 = ini_tokens_list[ini_token_idx]

                        # Simulate for max_sim_steps steps
                        HT.simulate_mf(x0, max_steps=max_sim_steps)
                    else:
                        # For the following betas, start from previous context in order to avoid falling into different
                        # attractors every time

                        HT.reset_data_keep_context()
                        HT.simulate_mf_from_context(max_steps=max_sim_steps)

                    for stat_name in stats_to_save_plot:
                        # Accumulate results in a var of beta_list length
                        results_beta_list[stat_name].append(np.copy(HT.mf_statistics[stat_name]))

                if reverse_betas:
                    for stat_name in stats_to_save_plot:
                        # Flip again the vector to keep the order
                        results_beta_list[stat_name] = np.flip(results_beta_list[stat_name])

                    beta_string = ("/min_beta-" + str(beta_list[-1]) + "-max_beta-" + str(beta_list[0]) +
                                   "-num_betas-" + str(len(beta_list)) + "-reverse_betas-keep_context-" +
                                   str(int(keep_context)))
                else:
                    beta_string = ("/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1]) +
                                   "-num_betas-" + str(len(beta_list)) + "-keep_context-" +
                                   str(int(keep_context)))

                if correlations_from_weights != 0:
                    gaussian_scale_name_str = ""
                else:
                    gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

                if save_non_transient == True:
                    save_non_transient_str = ""
                else:
                    save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"

                # Save/plot results for each ini_token, W config, and num_feat_patterns
                folder_path = ("results/infN-correlations_from_weights-" + str(correlations_from_weights)
                               + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                               + str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                               + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-"
                               + normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                               + "-num_segments_corrs-" + str(num_segments_corrs) + "-pe_mode-" + str(pe_mode)
                               + gaussian_scale_name_str + "/max_sim_steps-" + str(max_sim_steps)
                               + save_non_transient_str + "-context_size-" + str(context_size)
                               + beta_string + "/stats")

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                stats_data_path = folder_path + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".npz"
                np.savez_compressed(stats_data_path,
                                    mo_results_beta_list=results_beta_list["mo"],
                                    mo_se_results_beta_list=results_beta_list["mo_se"],
                                    mv_results_beta_list=results_beta_list["mv"],
                                    mq_results_beta_list=results_beta_list["mq"],
                                    mk_results_beta_list=results_beta_list["mk"],
                                    att_results_beta_list=results_beta_list["att"])

                print(f"Saved stats num_feat_patterns {num_feat_patterns}, seed {seed}, ini_token_idx {ini_token_idx}")


def plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps,
            max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str, reorder_weights,
            save_not_plot,
            stats_to_save_plot, correlations_from_weights, num_segments_corrs, pe_mode, se_per_contribution,
            keep_context,
            reverse_betas, gaussian_scale_str, save_non_transient):
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
                               + save_non_transient_str + "-context_size-" + str(context_size)
                               + "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1])
                               + "-num_betas-" + str(len(beta_list)) + f"{reverse_betas_str}-keep_context-"
                               + str(int(keep_context)))

                stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                                   + str(ini_token_idx) + ".npz")

                # Load data
                data = np.load(stats_data_path)

                show_max_num_patterns = 6

                # Load each stat and plot/save it
                for stat_name in stats_to_save_plot:

                    stat_results_beta_list = data[f"{stat_name}_results_beta_list"]

                    # Create folder if it does not exist and we are saving the image
                    if save_not_plot and (not os.path.exists(folder_path + f"/{stat_name}/")):
                        os.makedirs(folder_path + f"/{stat_name}/")

                    image_format = ".jpeg"

                    stat_save_path = (folder_path + f"/{stat_name}/seed-" + str(seed) + "-ini_token_idx-" +
                                      str(ini_token_idx) + "-transient_steps-" + str(
                                num_transient_steps) + image_format)

                    title = f"CORR_MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed}"

                    plot_bifurcation_diagram(stat_results_beta_list, beta_list, num_feat_patterns, stat_save_path,
                                             num_transient_steps_plot_arg, feat_name=stat_name,
                                             show_max_num_patterns=show_max_num_patterns, save_not_plot=save_not_plot,
                                             title=title)


if __name__ == "__main__":
    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    # Create variables for the Hopfield Transformer (HT)
    # seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 26]
    seed_list = [0]
    # seed_list = [0, 1, 2]
    # seed_list = [3, 4, 5]
    # seed_list = [6, 7]
    # seed_list = [8, 9]

    # seed_list = [0, 1, 2, 3, 4]
    # seed_list = [5, 6, 7, 8, 9]

    # seed_list = [18]

    # beta_list = np.linspace(0, 3, 1000)
    beta_list = np.linspace(1, 3, 2)
    se_per_contribution = tentative_semantic_embedding_size / (
                tentative_semantic_embedding_size + positional_embedding_size)
    # num_feat_patterns_list = [3, 2, 1]
    num_feat_patterns_list = [1]
    num_transient_steps = 10
    max_sim_steps = num_transient_steps + 30

    keep_context = False
    reverse_betas = False

    num_ini_tokens = 1
    reorder_weights = False
    normalize_weights_str = "np.sqrt(N)*M"
    correlations_from_weights = 3  # 0 use gaussian corrs, 1 create from weight matrices, 2 uniform means, 3 segments
    gaussian_scale = "0.5"  # Only applicable if correlations_from_weights=0
    pe_mode = 0
    num_segments_corrs = 3  # Only applicable if correlations_from_weights=3
    save_non_transient = False
    save_not_plot = False

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise ("You cannot discard more timesteps than you are simulating.")

    stats_to_save_plot = ["mo", "mo_se", "att"]
    # stats_to_save_plot = ["mo", "mo_se"]

    start = time.time()

    runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
           max_sim_steps,
           context_size, num_ini_tokens, seed_list, normalize_weights_str, reorder_weights, stats_to_save_plot,
           se_per_contribution, correlations_from_weights, num_segments_corrs, pe_mode, keep_context, reverse_betas,
           gaussian_scale, save_non_transient)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    ini_tokens_list = range(0, num_ini_tokens)
    plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps,
            max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str, reorder_weights,
            save_not_plot,
            stats_to_save_plot, correlations_from_weights, num_segments_corrs, pe_mode, se_per_contribution,
            keep_context,
            reverse_betas, gaussian_scale, save_non_transient)
