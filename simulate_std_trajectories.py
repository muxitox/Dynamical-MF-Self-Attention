import numpy as np
import matplotlib.pyplot as plt
from models.HopfieldTransformerPE import HopfieldTransformer
from models.HopfieldTransformerPE import Embedding
from plotting.plotting import plot_2_statistics

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

    label_tag = ["std", "mf"]
    for stat_name in HT.statistics_names:
        plot_2_statistics(mean_std_statistics[stat_name], HT.mf_statistics[stat_name], stat_name, num_feat_patterns,
                        num_plotting_steps, show_max_num_patterns=6)

    print("Done.")
