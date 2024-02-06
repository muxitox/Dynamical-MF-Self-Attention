import numpy as np
from bifurcation_diagrams import plotter


if __name__ == "__main__":
    # Instantiate vocabulary
    semantic_embedding_size = 100
    positional_embedding_size = 10

    # Create variables for the Hopfield Transformer (HT)
    seed_list = [1]
    beta_list = np.linspace(0, 3, 1000)
    num_feat_patterns_list = [16]
    num_transient_steps = 800
    max_sim_steps = 1024
    save_not_plot = False
    reorder_weights = False
    normalize_weights = True
    ini_tokens_list = [0]

    plotter(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
           max_sim_steps, ini_tokens_list, seed_list, normalize_weights, reorder_weights, save_not_plot)