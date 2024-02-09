import numpy as np
import matplotlib.pyplot as plt
from HopfieldTransformerPE_fullcontext import HopfieldTransformer as HopfieldTransformerFC
from HopfieldTransformerPE import HopfieldTransformer
from HopfieldTransformerPE import Embedding

def plot_statistics(stat1, stat2, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None):

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

        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, feat], label="std")
        local_ax.plot(num_plotting_steps_arange, stat2[:num_plotting_steps, feat], '--', label="mf")

        if feat > 3:
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
    context_size = 10
    max_sim_steps = 10

    # Create seed for reproducibility
    # Nice seed for reorder of W (no more constrains), 14 se spins 6 pe spins 6 features: 10. Sample = True Interesting cycle.
    # Seed 13 (8, 16 + 6) does not coincide with std model
    seed = 13

    np.random.seed(seed)

    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns, embedding_size=embedding_size,
                             vocab=vocab, max_sim_steps=max_sim_steps)

    # Select initial token
    random_idx = True
    if random_idx:
        x0_idx = 684  # You need to have an initial token to start decoding
        x0 = vocab.encode(x0_idx)
    else:
        x0 = HT.W[0]
        x0[semantic_embedding_size:] = -1  # Set embedding position for the first token to 0
        x0_idx = vocab.decode(x0)

        print(f"Initializing the model with the token with index {x0_idx}")

    print("List of tokens encoded in the features")
    print(HT.decoded_tokens)

    num_runs = 1

    mean_att = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mo_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mo_se_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mv_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mq_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mk_data = np.zeros((max_sim_steps, num_feat_patterns))

    for i in range(0, num_runs):
        # Instantiate HT with the above created vocabulary

        # print("Simulating standard Transformer...")
        HT.reset_data()
        selected_tokens = HT.simulate(x0, max_steps=max_sim_steps, verbose=False)

        num_diff_tokens = len(np.unique(selected_tokens[10:]))
        # if num_diff_tokens > 2:
        #     print("Seed:", seed, "Num different tokens: ", num_diff_tokens)

        #  Collect data to compute the mean of the trajectory
        mean_att += HT.att
        mean_mo_data += HT.mo_data
        mean_mo_se_data += HT.mo_se_data
        mean_mv_data += HT.mv_data
        mean_mq_data += HT.mq_data
        mean_mk_data += HT.mk_data

    # Compute mean
    mean_att /= num_runs
    mean_mo_data /= num_runs
    mean_mo_se_data /= num_runs
    mean_mv_data /= num_runs
    mean_mq_data /= num_runs
    mean_mk_data /= num_runs

    print("Simulating MF Transformer...")
    HT.simulate_mf(x0, max_steps=max_sim_steps)
    print("Done.")

    # Plotting
    print("Plotting statistics...")
    num_plotting_steps = max_sim_steps
    plot_statistics(mean_att, HT.att_mf, "Att", num_feat_patterns, num_plotting_steps, show_max_num_patterns=6)

    plot_statistics(mean_mo_data, HT.mo, "mo", num_feat_patterns, num_plotting_steps, show_max_num_patterns=6)

    plot_statistics(mean_mo_se_data, HT.mo_se, "mo_se", num_feat_patterns, num_plotting_steps, show_max_num_patterns=6)

    plot_statistics(mean_mv_data, HT.mv, "mv", num_feat_patterns, num_plotting_steps, show_max_num_patterns=6)

    plot_statistics(mean_mq_data, HT.mq, "mq", num_feat_patterns, num_plotting_steps, show_max_num_patterns=6)

    plot_statistics(mean_mk_data, HT.mk, "mk", num_feat_patterns, num_plotting_steps, show_max_num_patterns=6)
    print("Done.")
