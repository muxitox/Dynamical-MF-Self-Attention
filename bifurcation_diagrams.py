import numpy as np
from HopfieldTransformerPE import Embedding, HopfieldTransformer
import matplotlib.pyplot as plt
import os


def plot_bifurcation_diagram(mo_results_beta_list, beta_list, num_feat_patterns, save_path, feat_name, show_max_num_patterns=None):
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
            unique_values_feat = mo_results_beta_list[b_idx][:, feat]
            beta_values_feat = np.ones(len(unique_values_feat)) * beta_list[b_idx]

            feat_Y_values.extend(unique_values_feat)
            feat_X_values.extend(beta_values_feat)

        local_ax.plot(feat_X_values, feat_Y_values, ls='', marker=',')
        if feat_name != "att":
            local_ax.set_ylim(-1, 1)

        if feat > num_feat_patterns-2:
            local_ax.set_xlabel("beta")
        # local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    fig.suptitle(f"Bifurcation_diagram")
    fig.savefig(save_path)
    plt.close()


def runner(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, num_ini_tokens, seed_list):
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    # We don't initialize the vocab as it's more efficient to work without a dict with the MF implementation
    # vocab.initialize()

    np.random.seed(0)
    ini_tokens_idx_list = np.random.randint(2 ** semantic_embedding_size, size=num_ini_tokens)

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:

            np.random.seed(seed)

            # Initialize transformer weights and create variables for storing results
            HT = HopfieldTransformer(0, 0, num_feat_patterns=num_feat_patterns,
                                     embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps)

            for ini_token_idx in ini_tokens_idx_list:

                mo_results_beta_list = []
                mo_se_results_beta_list = []
                mv_results_beta_list = []
                mq_results_beta_list = []
                mk_results_beta_list = []
                att_results_beta_list = []

                for beta in beta_list:
                    beta_o = beta
                    beta_att = beta

                    # Reset the matrix for storing results
                    HT.set_betas(beta_o, beta_att)
                    HT.reset_data()
                    # Encode initial token with position 0
                    x0 = vocab.encode_force(ini_token_idx, 0)

                    # Simulate for max_sim_steps steps
                    HT.simulate_mf(x0, max_steps=max_sim_steps)

                    # Discard burnout steps
                    mo_result = np.copy(HT.mo[num_transient_steps:])
                    mo_se_result = np.copy(HT.mo_se[num_transient_steps:])
                    mv_result = np.copy(HT.mv[num_transient_steps:])
                    mq_result = np.copy(HT.mq[num_transient_steps:])
                    mk_result = np.copy(HT.mk[num_transient_steps:])
                    att_result = np.copy(HT.att_mf[num_transient_steps:])

                    # Accumulate results in a var of beta_list length
                    mo_results_beta_list.append(mo_result)
                    mo_se_results_beta_list.append(mo_se_result)
                    mv_results_beta_list.append(mv_result)
                    mq_results_beta_list.append(mq_result)
                    mk_results_beta_list.append(mk_result)
                    att_results_beta_list.append(att_result)

                # Save/plot results for each ini_token, W config, and num_feat_patterns
                folder_path = ("results/" + "se_size-" + str(semantic_embedding_size) + "-pe_size-" + str(
                    positional_embedding_size)
                               + "/num_feat_patterns-" + str(num_feat_patterns) + "/max_sim_steps-" + str(max_sim_steps)
                               + "-num_transient_steps_" + str(num_transient_steps) + "/min_beta-" + str(beta_list[0])
                               + "-max_beta-" + str(beta_list[-1]) + "-num_betas-" + str(len(beta_list)))

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                save_path = folder_path + "/mo-seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".png"
                plot_bifurcation_diagram(mo_results_beta_list, beta_list, num_feat_patterns, save_path, feat_name='mo',
                                         show_max_num_patterns=6)
                save_path = folder_path + "/mo_se-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".png"
                plot_bifurcation_diagram(mo_se_results_beta_list, beta_list, num_feat_patterns, save_path, feat_name='mo_se',
                                         show_max_num_patterns=6)
                save_path = folder_path + "/mv-seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".png"
                plot_bifurcation_diagram(mv_results_beta_list, beta_list, num_feat_patterns, save_path, feat_name='mv',
                                         show_max_num_patterns=6)
                save_path = folder_path + "/mq-seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".png"
                plot_bifurcation_diagram(mq_results_beta_list, beta_list, num_feat_patterns, save_path, feat_name='mq',
                                         show_max_num_patterns=6)
                save_path = folder_path + "/mk-seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".png"
                plot_bifurcation_diagram(mk_results_beta_list, beta_list, num_feat_patterns, save_path, feat_name='mk',
                                         show_max_num_patterns=6)
                save_path = folder_path + "/att-seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx) + ".png"
                plot_bifurcation_diagram(att_results_beta_list, beta_list, num_feat_patterns, save_path, feat_name='att',
                                         show_max_num_patterns=6)

                print(f"Plotted num_feat_patterns {num_feat_patterns}, seed {seed}, ini_token_idx {ini_token_idx}")


if __name__ == "__main__":
    # Instantiate vocabulary
    semantic_embedding_size = 10
    positional_embedding_size = 8

    # Create variables for the Hopfield Transformer (HT)
    seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    beta_list = np.linspace(0, 4, 200)
    num_feat_patterns_list = [1, 2, 4, 8, 16]
    num_transient_steps = 50
    max_sim_steps = 256
    num_ini_tokens = 3

    runner(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, num_ini_tokens, seed_list)
