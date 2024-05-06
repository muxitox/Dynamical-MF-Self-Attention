import numpy as np
from bifurcation_diagrams_inf import plotter

if __name__ == "__main__":

    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    # New
    # 1 pat: seed 2 0.25, 0.27
    # 2 pat: seed 18 0.8, 1.3
    # 3 pat: seed 1 0.35, 0.8

    # VARS FOR LOADING CHECKPOINTS
    # Create variables for the Hopfield Transformer (HT)
    seed_list = [1]
    beta_list = np.linspace(0, 4, 1500)
    # se_per_contribution_list = [tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)]
    se_per_contribution_list =[0.98]
    num_feat_patterns_list = [3]
    ini_tokens_list = [0]
    ini_token_from_w = 1
    num_transient_steps = 100000
    max_sim_steps = num_transient_steps + 20000
    keep_context = True
    reverse_betas = False

    # Specific stats for plotting
    # min_max_beta_to_show = [0.35, 0.4]
    min_max_beta_to_show = None

    # seed_list = [ 1]
    # num_feat_patterns_list = [3]

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise ("You cannot discard more timesteps than you are simulating.")

    reorder_weights = False
    normalize_weights_str_o = "N"
    normalize_weights_str_att = "N**2*np.sqrt(M)"
    scaling_o = 1
    scaling_att = 100
    compute_inf_normalization = True
    correlations_from_weights = 3  # 0 use gaussian corrs, 1 create from weight matrices, 2 uniform means, 3 segments
    num_segments_corrs = 3         # Only applicable if correlations_from_weights=3
    pe_mode = 0
    gaussian_scale = "0.5"         # Only applicable if correlations_from_weights=0
    save_non_transient = False
    save_not_plot = False

    # stats_to_save_plot = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    stats_to_save_plot = ["mo_se"]


    plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str_att,
            normalize_weights_str_o, reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights,
            num_segments_corrs, pe_mode, se_per_contribution_list, keep_context, reverse_betas, gaussian_scale,
            save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w,
            min_max_beta_to_show=min_max_beta_to_show)

