import numpy as np
import matplotlib.pyplot as plt
from models.HopfieldTransformerPE import HopfieldTransformer
from models.HopfieldTransformerPE import Embedding

def plot_statistics(stat1, stat2, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    nrows = (num_feat_patterns + 1) // 2

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, 4 * nrows), constrained_layout=True)

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
    positional_embedding_size = 4
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    beta = 4
    beta_o = beta
    beta_att = beta

    num_feat_patterns = 6
    context_size = 16
    max_sim_steps = 32
    normalize_weights_str = "np.sqrt(N*M)"
    # normalize_weights_str = "N"
    normalize_weights_str = normalize_weights_str.replace(" ", "")

    # Create seed for reproducibility
    # Nice seed for reorder of W (no more constrains), 14 se spins 6 pe spins 6 features: 10. Sample = True Interesting cycle.
    # Seed 13 (8, 16 + 6) does not coincide with std model
    seed = 27

    np.random.seed(seed)

    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns, embedding_size=embedding_size,
                             vocab=vocab, max_sim_steps=max_sim_steps, context_size=10, normalize_weights_str=normalize_weights_str)

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

    # Create variables to compute the mean
    mean_std_statistics = {}
    for stat_name in HT.statistics_names:
        mean_std_statistics[stat_name] = np.zeros((max_sim_steps, num_feat_patterns))

    for i in range(0, num_runs):
        # Instantiate HT with the above created vocabulary

        # print("Simulating standard Transformer...")
        HT.reset_data()
        selected_tokens = HT.simulate(x0, max_steps=max_sim_steps, verbose=True)

        num_diff_tokens = len(np.unique(selected_tokens[10:]))
        # if num_diff_tokens > 2:
        #     print("Seed:", seed, "Num different tokens: ", num_diff_tokens)

        #  Collect data to compute the mean of the trajectory
        mean_std_statistics["att"] += HT.std_statistics["att"]
        mean_std_statistics["mo"] += HT.std_statistics["mo"]
        mean_std_statistics["mo_se"] += HT.std_statistics["mo_se"]
        mean_std_statistics["mv"] += HT.std_statistics["mv"]
        mean_std_statistics["mq"] += HT.std_statistics["mq"]
        mean_std_statistics["mk"] += HT.std_statistics["mk"]

    # Compute mean
    for stat_name in HT.statistics_names:
        mean_std_statistics[stat_name] /= num_runs

    print("Simulating MF Transformer...")
    HT.simulate_mf(x0, max_steps=max_sim_steps)
    print("Done.")

    # Plotting
    print("Plotting statistics...")
    num_plotting_steps = max_sim_steps

    for stat_name in HT.statistics_names:
        plot_statistics(mean_std_statistics[stat_name], HT.mf_statistics[stat_name], stat_name, num_feat_patterns,
                        num_plotting_steps, show_max_num_patterns=6)

    print("Done.")
