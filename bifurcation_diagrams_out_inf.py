import numpy as np
from models.Embedding import Embedding
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from plotting.plotting import plot_bifurcation_diagram, plot_filtered_bifurcation_diagram
import os
import time


def create_pathname(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
                    num_transient_steps, max_sim_steps, context_size, normalize_weights_str_att,
                    normalize_weights_str_o, reorder_weights, se_per_contribution,
                    correlations_from_weights, num_segments_corrs, pe_mode, keep_context, reverse_betas, gaussian_scale_str,
                    save_non_transient, compute_inf_normalization, scaling_o, scaling_att, beta_att):

    if reverse_betas:
        beta_string = ("/beta_att-" + (beta_att) + "-min_beta_o-" + str(beta_list[-1]) + "-max_beta_o-"
                       + str(beta_list[0]) + "-num_betas-" + str(len(beta_list)) + "-reverse_betas-keep_context-" +
                       str(int(keep_context)))
    else:
        beta_string = ("/beta_att-" + str(beta_att) + "-min_beta_o-" + str(beta_list[0]) + "-max_beta_o-" +
                       str(beta_list[-1]) + "-num_betas-" + str(len(beta_list)) + "-keep_context-" +
                       str(int(keep_context)))

    if correlations_from_weights != 0:
        gaussian_scale_name_str = ""
    else:
        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

    if save_non_transient == True:
        save_non_transient_str = ""
    else:
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"

    if normalize_weights_str_o == normalize_weights_str_att:
        normalize_weights_name_str = "-normalize_weights-" + normalize_weights_str_att
    else:
        normalize_weights_name_str = ("-normalize_weights_att-" + normalize_weights_str_att +
                                      "-normalize_weights_o-" + normalize_weights_str_o)

    scaling_str = ""
    if scaling_o != 1:
        scaling_str += "-scaling_o-" + str(scaling_o)
    if scaling_att != 1:
        scaling_str += "-scaling_att-" + str(scaling_att)

    compute_inf_normalization_str = ""
    if compute_inf_normalization:
        compute_inf_normalization_str = "-inf_norm"

    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = ("results_out/infN-correlations_from_weights-" + str(correlations_from_weights)
                   + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                   + str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(reorder_weights)) + "-num_segments_corrs-" + str(num_segments_corrs)
                   + "-pe_mode-" + str(pe_mode) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(max_sim_steps) + save_non_transient_str + "-context_size-" + str(context_size)
                   + beta_string)

    return folder_path



def define_ini_token(ini_token_from_w, HT, ini_token_idx, ini_tokens_list):
    if ini_token_from_w == 0:
        # Encode initial token with position 0
        x0 = ini_tokens_list[ini_token_idx]
    elif ini_token_from_w == 1:
        x0 = HT.Wo[ini_token_idx]
    elif ini_token_from_w == 2:
        x0 = HT.Wv[ini_token_idx]
    elif ini_token_from_w == 3:
        x0 = HT.Wq[ini_token_idx]
    elif ini_token_from_w == 4:
        x0 = HT.Wk[ini_token_idx]

    return x0

def runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
           num_transient_steps, max_sim_steps, context_size, num_ini_tokens, seed_list, normalize_weights_str_att,
           normalize_weights_str_o, reorder_weights, stats_to_save_plot, se_per_contribution_list,
           correlations_from_weights, num_segments_corrs, pe_mode, keep_context, reverse_betas, gaussian_scale_str,
           save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_att):

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
            for se_per_contribution in se_per_contribution_list:

                np.random.seed(seed)

                # Initialize transformer weights and create variables for storing results
                HT = HopfieldTransformerInfN(0, 0, num_feat_patterns=num_feat_patterns,
                                             positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                             context_size=context_size, max_sim_steps=max_sim_steps,
                                             min_saved_step=min_saved_step,
                                             normalize_weights_str_att=normalize_weights_str_att,
                                             normalize_weights_str_o=normalize_weights_str_o,
                                             reorder_weights=reorder_weights,
                                             correlations_from_weights=correlations_from_weights,
                                             num_segments_corrs=num_segments_corrs, pe_mode=pe_mode,
                                             semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                             se_per_contribution=se_per_contribution,
                                             compute_inf_normalization=compute_inf_normalization,
                                             N_normalization=9999,
                                             scaling_o=scaling_o,
                                             scaling_att=scaling_att)

                for ini_token_idx in range(0, num_ini_tokens):

                    # Initialize structure for saving the results for each beta
                    results_beta_list = {}
                    for stat_name in HT.statistics_names:
                        results_beta_list[stat_name] = []

                    for beta_idx in range(0, len(beta_list)):
                        beta_o = beta_list[beta_idx]
                        beta_att = beta_att
                        HT.set_betas(beta_o, beta_att)

                        print(f"Computing seed {seed} beta {beta_idx}/{len(beta_list)}", flush=True)

                        if beta_idx == 0 or not keep_context:
                            # For the first beta in the series reset everything and start from scratch
                            # Reset the matrix for storing results
                            HT.reset_data()

                            x0 = define_ini_token(ini_token_from_w, HT, ini_token_idx, ini_tokens_list)
                            if ini_token_from_w != 0:
                                x0[-positional_embedding_size:] = -1  # Initialize position to -1

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

                    folder_path = create_pathname(num_feat_patterns, tentative_semantic_embedding_size,
                                                  positional_embedding_size, beta_list, num_transient_steps,
                                                  max_sim_steps, context_size, normalize_weights_str_att,
                                                  normalize_weights_str_o, reorder_weights, se_per_contribution,
                                                  correlations_from_weights, num_segments_corrs, pe_mode, keep_context,
                                                  reverse_betas, gaussian_scale_str, save_non_transient,
                                                  compute_inf_normalization, scaling_o, scaling_att, beta_att)

                    folder_path = folder_path + "/stats"

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)


                    ini_token_mode_str = ""
                    if ini_token_from_w != 0:
                        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
                    stats_data_path = (folder_path + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx)
                                       + ini_token_mode_str + ".npz")

                    print("Saving results in ", os.path.abspath(stats_data_path))
                    np.savez_compressed(stats_data_path,
                                        mo_results_beta_list=results_beta_list["mo"],
                                        mo_se_results_beta_list=results_beta_list["mo_se"],
                                        mv_results_beta_list=results_beta_list["mv"],
                                        mq_results_beta_list=results_beta_list["mq"],
                                        mk_results_beta_list=results_beta_list["mk"],
                                        att_results_beta_list=results_beta_list["att"])

                    print(f"Saved stats num_feat_patterns {num_feat_patterns}, seed {seed}, ini_token_idx {ini_token_idx}")


def plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str_att,
            normalize_weights_str_o, reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights,
            num_segments_corrs, pe_mode, se_per_contribution_list, keep_context, reverse_betas, gaussian_scale_str,
            save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_o,
            filtering_range, min_max_beta_to_show=None, show_title=False):

    if min_max_beta_to_show is None:
        min_beta_idx = 0
        max_beta_idx = None
    else:
        min_beta_idx = np.searchsorted(beta_list, min_max_beta_to_show[0])
        max_beta_idx = np.searchsorted(beta_list, min_max_beta_to_show[1]) + 1

    if save_non_transient == True:
        num_transient_steps_plot_arg = num_transient_steps
    else:
        num_transient_steps_plot_arg = 0

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:
            for se_per_contribution in se_per_contribution_list:
                for ini_token_idx in ini_tokens_list:

                    folder_path = create_pathname(num_feat_patterns, tentative_semantic_embedding_size,
                                                  positional_embedding_size, beta_list, num_transient_steps,
                                                  max_sim_steps, context_size, normalize_weights_str_att,
                                                  normalize_weights_str_o, reorder_weights, se_per_contribution,
                                                  correlations_from_weights, num_segments_corrs, pe_mode, keep_context,
                                                  reverse_betas, gaussian_scale_str, save_non_transient,
                                                  compute_inf_normalization, scaling_o, scaling_att, beta_o)

                    ini_token_mode_str = ""
                    if ini_token_from_w != 0:
                        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
                    stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                                       + str(ini_token_idx) + ini_token_mode_str + ".npz")

                    # Load data
                    data = np.load(stats_data_path)

                    show_max_num_patterns = 6

                    # Load each stat and plot/save it
                    for stat_name in stats_to_save_plot:

                        stat_results_beta_list = data[f"{stat_name}_results_beta_list"]

                        # Create folder if it does not exist and we are saving the image
                        if save_not_plot and (not os.path.exists(folder_path + f"/{stat_name}/")):
                            os.makedirs(folder_path + f"/{stat_name}/")

                        # image_format = ".jpeg"
                        image_format = ".pdf"

                        stat_save_path = (folder_path + f"/{stat_name}/seed-" + str(seed) + "-ini_token_idx-" +
                                          str(ini_token_idx) + "-transient_steps-" + str(
                                    num_transient_steps) + image_format)

                        if show_title:
                            title = f"CORR_MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed}"
                        else:
                            title = None

                        plot_bifurcation_diagram(stat_results_beta_list[min_beta_idx:max_beta_idx],
                                                 beta_list[min_beta_idx:max_beta_idx], num_feat_patterns, stat_save_path,
                                                 num_transient_steps_plot_arg, feat_name=stat_name,
                                                 show_max_num_patterns=show_max_num_patterns, save_not_plot=save_not_plot,
                                                 title=title)

                        for filter_idx in range(num_feat_patterns):

                            if show_title:
                                title = (
                                    f"CORRm={correlations_from_weights} CTX={context_size} NUM_PAT={num_feat_patterns} "
                                    f"SEED={seed} Filter={filtering_range}")
                            else:
                                title = None

                            filtered_save_path = (folder_path + f"/{stat_name}/seed-" + str(seed) + "-ini_token_idx-" +
                                                  str(ini_token_idx) + "-transient_steps-" + str(
                                        num_transient_steps) + "-filter_idx-" + str(filter_idx) +
                                                  "-filter_rg-" + str(filtering_range) + image_format)

                            plot_filtered_bifurcation_diagram(stat_results_beta_list[min_beta_idx:max_beta_idx],
                                                              stat_results_beta_list[min_beta_idx:max_beta_idx],
                                                              filter_idx,
                                                              beta_list[min_beta_idx:max_beta_idx], num_feat_patterns,
                                                              filtered_save_path,
                                                              num_transient_steps_plot_arg,
                                                              filtering_range=filtering_range,
                                                              feat_name=stat_name,
                                                              show_max_num_patterns=show_max_num_patterns,
                                                              save_not_plot=save_not_plot,
                                                              title=title)


if __name__ == "__main__":
    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    # Create variables for the Hopfield Transformer (HT)
    # seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 26]

    # New
    # 1 pat: seed 2 0.25, 0.27
    # 2 pat: seed 18 0.8, 1.3
    # 2 pat: seed 10. pe 2: beta 2.2 - 2.8, pe:4 1.5 - 2.2
    # 3 pat: seed 1 (0.35,0.3) - 0.8, 0.37 - 0.45

    beta_att = 2.2
    beta_list = np.linspace(0, 3, 1000)
    # beta_list = np.linspace(1.25, 2.4, 1000)
    # beta_list = np.linspace(1.75, 2.75, 1200)
    # se_per_contribution_list = [(tentative_semantic_embedding_size /
    #                        (tentative_semantic_embedding_size + positional_embedding_size))]
    se_per_contribution_list = [0.98]

    # seed_list = [1, 2, 4, 6]
    # seed_list = [10, 14, 18]
    seed_list = [1]
    num_feat_patterns_list = [3]
    num_transient_steps = 100000
    max_sim_steps = num_transient_steps + 20000

    keep_context = False
    reverse_betas = False

    num_ini_tokens = 1
    ini_token_from_w = 1
    reorder_weights = False
    normalize_weights_str_o = "N"
    normalize_weights_str_att = "N**2*np.sqrt(M)"
    scaling_o = 1
    scaling_att = 100
    filtering_range = 0.001
    compute_inf_normalization = True
    correlations_from_weights = 3  # 0 use gaussian corrs, 1 create from weight matrices, 2 uniform means, 3 segments
    gaussian_scale = "0.5"  # Only applicable if correlations_from_weights=0
    pe_mode = 0
    num_segments_corrs = 3  # Only applicable if correlations_from_weights=3
    save_non_transient = False
    save_not_plot = False
    show_title = True

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise ("You cannot discard more timesteps than you are simulating.")

    # stats_to_save_plot = ["mo", "mo_se", "att"]
    stats_to_save_plot = ["mo_se"]

    start = time.time()

    runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
           num_transient_steps, max_sim_steps, context_size, num_ini_tokens, seed_list, normalize_weights_str_att,
           normalize_weights_str_o, reorder_weights, stats_to_save_plot, se_per_contribution_list,
           correlations_from_weights, num_segments_corrs, pe_mode, keep_context, reverse_betas, gaussian_scale,
           save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_att)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    ini_tokens_list = range(0, num_ini_tokens)
    plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str_att,
            normalize_weights_str_o, reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights,
            num_segments_corrs, pe_mode, se_per_contribution_list, keep_context, reverse_betas, gaussian_scale,
            save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_att,
            filtering_range, show_title=show_title)
