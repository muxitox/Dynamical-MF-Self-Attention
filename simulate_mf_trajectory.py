import numpy as np
from models.HopfieldTransformerMFPE import HopfieldTransformerMFPE
from models.Embedding import Embedding
from plotting.plotting import plot_save_statistics

if __name__ == "__main__":

    # Instantiate vocabulary
    semantic_embedding_size = 80
    positional_embedding_size = 4
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    # vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    beta = 0.3
    beta_o = beta
    beta_att = beta

    num_feat_patterns = 4
    max_sim_steps = 1000
    context_size = 16

    normalize_weights_str = "np.sqrt(N*M)"
    reorder_weights = False

    # Select initial token with seed 0
    np.random.seed(0)
    num_ini_tokens = 1
    ini_tokens_list = np.random.randint(2, size=(num_ini_tokens, embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    ini_token_idx = 0
    x0 = ini_tokens_list[ini_token_idx, :]

    # Create seed for reproducibility
    seed = 1
    np.random.seed(seed)

    HT = HopfieldTransformerMFPE(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                 embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps, context_size=context_size,
                                 normalize_weights_str=normalize_weights_str, reorder_weights=reorder_weights)

    num_runs = 1

    print("Simulating MF Transformer...")
    HT.simulate(x0, max_steps=max_sim_steps)
    print("Done.")

    # Plotting
    print("Plotting statistics...")
    num_plotting_steps = max_sim_steps

    for stat_name in HT.statistics_names:
        plot_save_statistics(HT.mf_statistics[stat_name], stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=6)
    print("Done.")
