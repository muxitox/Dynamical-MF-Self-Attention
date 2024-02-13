import numpy as np
from HopfieldTransformerPE import Embedding, HopfieldTransformer
import matplotlib.pyplot as plt
import os
import time


def plot_bifurcation_diagram(mo_results_beta_list, beta_list, num_feat_patterns, save_path, num_transient_steps,
                             feat_name, show_max_num_patterns=None, save_not_plot=True):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    nrows = (num_feat_patterns + 1) // 2

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(8, 2 * nrows), constrained_layout=True)

    for feat in range(0, num_feat_patterns):

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        else:
            local_ax = ax[row, feat % 2]

        feat_Y_values = []
        feat_X_values = []

        for b_idx in range(0, len(beta_list)):
            unique_values_feat = mo_results_beta_list[b_idx][num_transient_steps:, feat]
            beta_values_feat = np.ones(len(unique_values_feat)) * beta_list[b_idx]

            feat_Y_values.extend(unique_values_feat)
            feat_X_values.extend(beta_values_feat)

        local_ax.plot(feat_X_values, feat_Y_values, ls='', marker='.', ms='0.05')
        if feat_name != "att":
            local_ax.set_ylim(-1, 1)

        if feat > num_feat_patterns-2:
            local_ax.set_xlabel("beta")
        # local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    fig.suptitle(f"Bifurcation_diagram {feat_name}")
    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()


def runner(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, max_sim_steps,
           context_size, num_ini_tokens, seed_list, normalize_weights_str, reorder_weights, stats_to_save_plot):
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    # We don't initialize the vocab as it's more efficient to work without a dict with the MF implementation
    # vocab.initialize()

    np.random.seed(0)
    ini_tokens_list = np.random.randint(2, size=(num_ini_tokens, semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:

            np.random.seed(seed)

            # Initialize transformer weights and create variables for storing results
            HT = HopfieldTransformer(0, 0, num_feat_patterns=num_feat_patterns,
                                     embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps,
                                     context_size=context_size, normalize_weights_str=normalize_weights_str,
                                     reorder_weights=reorder_weights)

            for ini_token_idx in range(0, num_ini_tokens):

                # Initialize structure for saving the results for each beta
                results_beta_list = {}
                for stat_name in stats_to_save_plot:
                    results_beta_list[stat_name] = []

                for beta in beta_list:
                    beta_o = beta
                    beta_att = beta

                    # Reset the matrix for storing results
                    HT.set_betas(beta_o, beta_att)
                    HT.reset_data()
                    # Encode initial token with position 0
                    x0 = ini_tokens_list[ini_token_idx]

                    # Simulate for max_sim_steps steps
                    HT.simulate_mf(x0, max_steps=max_sim_steps)

                    for stat_name in HT.statistics_names:
                        # Accumulate results in a var of beta_list length
                        results_beta_list[stat_name].append(np.copy(HT.mf_statistics[stat_name]))

                # Save/plot results for each ini_token, W config, and num_feat_patterns
                folder_path = ("results/" + "se_size-" + str(semantic_embedding_size) + "-pe_size-" +
                               str(positional_embedding_size)
                               + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-" +
                               normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                               + "/max_sim_steps-" + str(max_sim_steps) + "-context_size-" + str(context_size) +
                               "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1]) +
                               "-num_betas-" + str(len(beta_list)) + "/stats")

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


def plotter(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str, reorder_weights, save_not_plot,
            stats_to_save_plot):

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:
            for ini_token_idx in ini_tokens_list:

                # Create folder path name as defined in the runner function
                folder_path = ("results/" + "se_size-" + str(semantic_embedding_size) + "-pe_size-" +
                               str(positional_embedding_size)
                               + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-" +
                               normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                               + "/max_sim_steps-" + str(max_sim_steps) + "-context_size-" + str(context_size) +
                               "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1]) +
                               "-num_betas-" + str(len(beta_list)))

                stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) +
                                   ".npz")

                # Load data
                data = np.load(stats_data_path)

                show_max_num_patterns = 6

                # Load each stat and plot/save it
                for stat_name in stats_to_save_plot:

                    stat_results_beta_list = data[f"{stat_name}_results_beta_list"]

                    # Create folder if it does not exist and we are saving the image
                    if save_not_plot and (not os.path.exists(folder_path + f"/{stat_name}/")):
                        os.makedirs(folder_path + f"/{stat_name}/")

                    stat_save_path = (folder_path + f"/{stat_name}/seed-" + str(seed) + "-ini_token_idx-" +
                                      str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + ".png")

                    plot_bifurcation_diagram(stat_results_beta_list, beta_list, num_feat_patterns, stat_save_path,
                                             num_transient_steps, feat_name=stat_name,
                                             show_max_num_patterns=show_max_num_patterns, save_not_plot=save_not_plot)




if __name__ == "__main__":
    # Instantiate vocabulary
    semantic_embedding_size = 100
    positional_embedding_size = 11

    # Create variables for the Hopfield Transformer (HT)
    seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    beta_list = np.linspace(0, 4, 1500)
    num_feat_patterns_list = [1, 16]
    # num_feat_patterns_list = [4, 6]
    # num_feat_patterns_list = [2, 10]
    num_transient_steps = 768
    max_sim_steps = 1536

    if max_sim_steps > 2**positional_embedding_size:
        raise("The positional embedding cannot simulate that many steps.")
    if num_transient_steps > max_sim_steps:
        raise("You cannot discard more timesteps than you are simulating.")

    context_size = 10
    num_ini_tokens = 3
    reorder_weights = False
    normalize_weights_str = "N"
    save_not_plot = True

    stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]

    start = time.time()

    runner(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, max_sim_steps,
           context_size, num_ini_tokens, seed_list, normalize_weights_str, reorder_weights, stats_to_save_plot)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time/60)
    print("elapsed time in hours", elapsed_time/3600)

    ini_tokens_list = range(0, num_ini_tokens)
    plotter(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
            max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str, reorder_weights, save_not_plot,
            stats_to_save_plot)
