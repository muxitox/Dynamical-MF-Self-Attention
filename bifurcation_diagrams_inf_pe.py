import numpy as np
from models.HopfieldTransformerPE import Embedding
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from plotting.plotting import plot_bifurcation_diagram
import os
import time


def runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, context_size, num_ini_tokens, seed_list, normalize_weights_str, reorder_weights,
           stats_to_save_plot, se_per_contribution_list, correlations_from_weights, num_segments_corrs, pe_mode,
           keep_context, reverse_se_per, gaussian_scale_str, save_non_transient, compute_inf_normalization):

    vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size)

    np.random.seed(0)
    ini_tokens_list = np.random.randint(2, size=(
    num_ini_tokens, tentative_semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    if reverse_se_per:  # Flip vector to avoid falling in the same fixed points due to the mean-field 0 a low beta
        beta_list = np.flip(beta_list)

    min_saved_step = 0
    if not save_non_transient:
        min_saved_step = num_transient_steps

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:
            for beta in beta_list:
                beta_o = beta
                beta_att = beta

                np.random.seed(seed)

                # Initialize transformer weights and create variables for storing results
                HT = HopfieldTransformerInfN(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                             positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                             context_size=context_size, max_sim_steps=max_sim_steps,
                                             min_saved_step=min_saved_step,
                                             normalize_weights_str=normalize_weights_str,
                                             reorder_weights=reorder_weights,
                                             correlations_from_weights=correlations_from_weights,
                                             num_segments_corrs=num_segments_corrs, pe_mode=pe_mode,
                                             semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                             se_per_contribution=1,
                                             compute_inf_normalization=compute_inf_normalization,
                                             N_normalization=9999)

                for ini_token_idx in range(0, num_ini_tokens):

                    # Initialize structure for saving the results for each beta
                    results_se_per_list = {}
                    for stat_name in HT.statistics_names:
                        results_se_per_list[stat_name] = []

                    for se_per_idx in range(0, len(se_per_contribution_list)):

                        se_per_contribution = se_per_contribution_list[se_per_idx]
                        HT.set_se_per_contribution(se_per_contribution)

                        print(f"Computing seed {seed} se_per {se_per_idx}/{len(se_per_contribution_list)}", flush=True)

                        if se_per_idx == 0 or not keep_context:
                            # For the first se_per in the series reset everything and start from scratch
                            # Reset the matrix for storing results
                            HT.reset_data()
                            # Encode initial token with position 0
                            x0 = ini_tokens_list[ini_token_idx]

                            # Simulate for max_sim_steps steps
                            HT.simulate_mf(x0, max_steps=max_sim_steps)
                        else:
                            # For the following se_pers, start from previous context in order to avoid falling into different
                            # attractors every time

                            HT.reset_data_keep_context()
                            HT.simulate_mf_from_context(max_steps=max_sim_steps)

                        for stat_name in stats_to_save_plot:
                            # Accumulate results in a var of beta_list length
                            results_se_per_list[stat_name].append(np.copy(HT.mf_statistics[stat_name]))

                    if reverse_se_per:
                        for stat_name in stats_to_save_plot:
                            # Flip again the vector to keep the order
                            results_se_per_list[stat_name] = np.flip(results_se_per_list[stat_name])

                        se_per_string = ("/min_se_per-" + str(se_per_contribution_list[-1]) + "-max_se_per-"
                                         + str(se_per_contribution_list[0]) + "-num_pes-" + str(len(beta_list))
                                         + "-reverse_se_per-keep_context-" + str(int(keep_context)))
                    else:
                        se_per_string = ("/min_se_per-" + str(se_per_contribution_list[0]) + "-max_se_per-" +
                                         str(se_per_contribution_list[-1]) + "-num_pes-" + str(len(se_per_contribution_list)) +
                                         "-keep_context-" + str(int(keep_context)))

                    if correlations_from_weights != 0:
                        gaussian_scale_name_str = ""
                    else:
                        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

                    if save_non_transient == True:
                        save_non_transient_str = ""
                    else:
                        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"

                    compute_inf_normalization_str = ""
                    if compute_inf_normalization:
                        compute_inf_normalization_str = "-inf_norm"

                    # Save/plot results for each ini_token, W config, and num_feat_patterns
                    folder_path = ("results/se_per/infN-correlations_from_weights-" + str(correlations_from_weights)
                                   + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                                   + str(positional_embedding_size) + "-beta-" + str(beta)
                                   + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-"
                                   + normalize_weights_str + compute_inf_normalization_str + "-reorder_weights-" +
                                   str(int(reorder_weights)) + "-num_segments_corrs-" + str(num_segments_corrs)
                                   + "-pe_mode-" + str(pe_mode) + gaussian_scale_name_str + "/max_sim_steps-"
                                   + str(max_sim_steps) + save_non_transient_str + "-context_size-" + str(context_size)
                                   + se_per_string + "/stats")

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    stats_data_path = folder_path + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".npz"
                    print(stats_data_path)
                    np.savez_compressed(stats_data_path,
                                        mo_results_se_per_list=results_se_per_list["mo"],
                                        mo_se_results_se_per_list=results_se_per_list["mo_se"],
                                        mv_results_se_per_list=results_se_per_list["mv"],
                                        mq_results_se_per_list=results_se_per_list["mq"],
                                        mk_results_se_per_list=results_se_per_list["mk"],
                                        att_results_se_per_list=results_se_per_list["att"])

                    print(f"Saved stats num_feat_patterns {num_feat_patterns}, seed {seed}, ini_token_idx {ini_token_idx}")


def plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str,
            reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights, num_segments_corrs,
            pe_mode, se_per_contribution_list, keep_context, reverse_betas, gaussian_scale_str, save_non_transient,
            compute_inf_normalization, min_max_se_per_to_show=None, show_title=False):

    reverse_se_per_str = ""
    if reverse_betas:
        reverse_se_per_str = "-reverse_se_per"

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

    compute_inf_normalization_str = ""
    if compute_inf_normalization:
        compute_inf_normalization_str = "-inf_norm"

    if min_max_se_per_to_show is None:
        min_se_per_idx = 0
        max_se_per_idx = None
    else:
        min_se_per_idx = np.searchsorted(se_per_contribution_list, min_max_se_per_to_show[0])
        max_se_per_idx = np.searchsorted(se_per_contribution_list, min_max_se_per_to_show[1]) + 1

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:
            for beta in beta_list:
                for ini_token_idx in ini_tokens_list:

                    folder_path = ("results/se_per/infN-correlations_from_weights-" + str(correlations_from_weights)
                                   + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                                   + str(positional_embedding_size) + "-beta-" + str(beta)
                                   + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-"
                                   + normalize_weights_str + compute_inf_normalization_str +
                                   "-reorder_weights-" + str(int(reorder_weights))
                                   + "-num_segments_corrs-" + str(num_segments_corrs) + "-pe_mode-" + str(pe_mode)
                                   + gaussian_scale_name_str + "/max_sim_steps-" + str(max_sim_steps)
                                   + save_non_transient_str + "-context_size-" + str(context_size)
                                   + "/min_se_per-" + str(se_per_contribution_list[0]) + "-max_se_per-" + str(se_per_contribution_list[-1])
                                   + "-num_pes-" + str(len(se_per_contribution_list)) + f"{reverse_se_per_str}-keep_context-"
                                   + str(int(keep_context)))

                    stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                                       + str(ini_token_idx) + ".npz")

                    # Load data
                    data = np.load(stats_data_path)

                    show_max_num_patterns = 6

                    # Load each stat and plot/save it
                    for stat_name in stats_to_save_plot:

                        stat_results_se_per_list = data[f"{stat_name}_results_se_per_list"]

                        # Create folder if it does not exist and we are saving the image
                        if save_not_plot and (not os.path.exists(folder_path + f"/{stat_name}/")):
                            os.makedirs(folder_path + f"/{stat_name}/")

                        image_format = ".jpeg"

                        stat_save_path = (folder_path + f"/{stat_name}/seed-" + str(seed) + "-ini_token_idx-" +
                                          str(ini_token_idx) + "-transient_steps-" + str(
                                    num_transient_steps) + image_format)

                        if show_title:
                            title = f"CORR_MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed}"
                        else:
                            title = None

                        plot_bifurcation_diagram(stat_results_se_per_list[min_se_per_idx:max_se_per_idx],
                                                 se_per_contribution_list[min_se_per_idx:max_se_per_idx], num_feat_patterns, stat_save_path,
                                                 num_transient_steps_plot_arg, feat_name=stat_name,
                                                 show_max_num_patterns=show_max_num_patterns, save_not_plot=save_not_plot,
                                                 title=title, is_beta=False)


if __name__ == "__main__":
    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 4
    context_size = 2 ** positional_embedding_size

    # Create variables for the Hopfield Transformer (HT)
    # seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 26]

    # New
    # 1 pat: seed 2 0.25, 0.27
    # 2 pat: seed 18 0.8, 1.3
    # 2 pat: seed 10. pe 2: beta 2.2 - 2.8, pe:4 1.5 - 2.2
    # 3 pat: seed 1 (0.35,0.3) - 0.8, 0.37 - 0.45


    # beta_list = np.linspace(0, 3, 1000)
    beta_list = [2]
    se_per_contribution_list = 1 - np.linspace(0, 0.5, 10)

    seed_list = [1]
    num_feat_patterns_list = [3]
    num_transient_steps = 100
    max_sim_steps = num_transient_steps + 400

    keep_context = True
    reverse_betas = False

    num_ini_tokens = 1
    reorder_weights = False
    normalize_weights_str = "np.sqrt(N)*M"
    compute_inf_normalization = True
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
           num_transient_steps, max_sim_steps, context_size, num_ini_tokens, seed_list, normalize_weights_str, reorder_weights,
           stats_to_save_plot, se_per_contribution_list, correlations_from_weights, num_segments_corrs, pe_mode,
           keep_context, reverse_betas, gaussian_scale, save_non_transient, compute_inf_normalization)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    ini_tokens_list = range(0, num_ini_tokens)
    plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str,
            reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights, num_segments_corrs, pe_mode,
            se_per_contribution_list, keep_context, reverse_betas, gaussian_scale, save_non_transient,
            compute_inf_normalization)
