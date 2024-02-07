import numpy as np
import matplotlib.pyplot as plt
from HopfieldTransformerPE import HopfieldTransformer
from HopfieldTransformerPE import Embedding
def plot_statistics(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    nrows = (num_feat_patterns + 1) // 2

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(8, 2 * nrows), constrained_layout=True)

    num_plotting_steps_arange = np.arange(num_plotting_steps)

    for feat in range(0, num_feat_patterns):

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        else:
            local_ax = ax[row, feat % 2]

        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, feat], label="mf")
        if feat > num_feat_patterns-2:
            local_ax.set_xlabel("t")
        local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    fig.suptitle(f"Evolution of {stat_name}")
    plt.show()

if __name__ == "__main__":


    # Instantiate vocabulary
    semantic_embedding_size = 14
    positional_embedding_size = 8
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    beta = 4
    beta_o = beta
    beta_att = beta

    num_feat_patterns = 6
    max_sim_steps = 200

    normalize_weights = True
    reorder_weights = True

    # Select initial token with seed 0
    np.random.seed(0)
    num_ini_tokens = 3
    ini_tokens_list = np.random.randint(2, size=(num_ini_tokens, embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    ini_token_idx = 0
    x0 = ini_tokens_list[ini_token_idx, :]

    # Create seed for reproducibility
    seed = 13
    np.random.seed(seed)

    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                     embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps,
                                     normalize_weights=normalize_weights, reorder_weights=reorder_weights)

    num_runs = 1

    print("Simulating MF Transformer...")
    HT.simulate_mf(x0, max_steps=max_sim_steps)
    print("Done.")

    # Plotting
    print("Plotting statistics...")
    num_plotting_steps = max_sim_steps
    plot_statistics(HT.att_mf, "Att", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mo, "mo", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mo_se, "mo_se", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mv, "mv", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mq, "mq", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mk, "mk", num_feat_patterns, num_plotting_steps)
    print("Done.")
