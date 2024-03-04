import numpy as np
from bifurcation_diagrams import plotter


if __name__ == "__main__":
    # Instantiate vocabulary
    semantic_embedding_size = 100
    positional_embedding_size = 11
    context_size = 10

    # Create variables for the Hopfield Transformer (HT)
    seed_list = [1]
    beta_list = np.linspace(0, 0.2, 1500)
    num_feat_patterns_list = [16]
    num_transient_steps = 1000
    max_sim_steps = 1536
    save_not_plot = False
    reorder_weights = False
    normalize_weights_str = "N"
    # normalize_weights_str = "np.sqrt(N+M)"
    ini_tokens_list = [0]
    keep_context = False

    stats_to_save_plot = ["mo_se"]

    plotter(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size, beta_list, num_transient_steps,
            max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str, reorder_weights,
            save_not_plot,
            stats_to_save_plot, keep_context)
