import numpy as np
import time
from bifurcation_diagrams_inf import runner, plotter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Specify the seed for the RNG", type=int)
parser.add_argument("--num_feat_patterns", help="Specify the number of features/patterns", type=int)
parser.add_argument("--tentative_semantic_embedding_size", help="Specify the tentative semantic emmbedding "
                                                                "size for creating the matrices in order to compute the correlations", type=int)
parser.add_argument("--positional_embedding_size", help="Specify the PE size", type=int)
parser.add_argument("--num_transient_steps", help="Specify the number transient steps to discard", type=int)
parser.add_argument("--max_sim_steps", help="Specify the number max steps to simulate", type=int)
parser.add_argument("--keep_context", help="Specify whether to keep the context from one exec to another",
                    type=bool, default=False)
parser.add_argument("--reverse_betas", help="Specify whether to reverse the order of the betas",
                    type=bool, default=False)
parser.add_argument("--reorder_weights", help="Specify whether to create the weight matrices from a reorder of a first one",
                    type=bool, default=False)
parser.add_argument("--num_ini_tokens", help="Specify number of initial tokens",
                    type=int, default=1)
parser.add_argument("--compute_inf_normalization", help="Specify whether to compute the true inf normalization",
                    type=bool, default=True)
parser.add_argument("--normalize_weights_str", help="Specify whether the normalization factor",
                    type=str, default="N*np.sqrt(M)")
parser.add_argument("--correlations_from_weights", help="Specify whether the correlations mode",
                    type=int, default=3)
parser.add_argument("--pe_mode", help="Specify the PE mode",
                    type=int, default=0)
parser.add_argument("--gaussian_scale", help="Specify the Gaussian scale (only applicable if correlations_from_weights==0)",
                    type=float, default=0.5)
parser.add_argument("--num_segments_corrs", help="Specify number of segments when correlations_from_weights==3",
                    type=int, default=0)
parser.add_argument("--save_non_transient", help="Specify whether to save the transient steps",
                    type=bool, default=False)
parser.add_argument("--min_beta", help="Specify the min beta value",
                    type=float, default=3)
parser.add_argument("--max_beta", help="Specify the max beta value",
                    type=float, default=3)
parser.add_argument("--num_betas", help="Specify the number of beta values",
                    type=int, default=3)
parser.add_argument("--min_pe", help="Specify the min pe value",
                    type=float, default=3)
parser.add_argument("--max_pe", help="Specify the max pe value",
                    type=float, default=3)
parser.add_argument("--num_pes", help="Specify the number of pe values",
                    type=int, default=3)
parser.add_argument("--pe_proportion_from_size", help="Specify how to compute the pe importance",
                    type=bool, default=True)
parser.add_argument("--save_not_plot", help="Specify the number of pe values",
                    type=bool, default=True)


if __name__ == "__main__":

    args = parser.parse_args()
    # Instantiate vocabulary
    tentative_semantic_embedding_size = args.tentative_semantic_embedding_size
    positional_embedding_size = args.positional_embedding_size
    context_size = 2 ** positional_embedding_size

    # Create variables for the Hopfield Transformer (HT)
    # seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 26]

    # New
    # 1 pat: seed 2 0.25, 0.27
    # 2 pat: seed 18 0.8, 1.3
    # 2 pat: seed 10. pe 2: beta 2.2 - 2.8, pe:4 1.5 - 2.2
    # 3 pat: seed 1 (0.35,0.3) - 0.8, 0.37 - 0.45


    # beta_list = np.linspace(0, 3, 1000)
    beta_list = np.linspace(args.min_beta, args.max_beta, args.num_betas)
    if args.pe_proportion_from_size:
        se_per_contribution = [(tentative_semantic_embedding_size /
                           (tentative_semantic_embedding_size + positional_embedding_size))]
    else:
        se_per_contribution = [1 - np.linspace(args.min_pe, args.max_pe, args.num_pe)]

    seed_list = [args.seed]
    num_feat_patterns_list = [args.num_feat_patterns]
    num_transient_steps = args.num_transient_steps
    max_sim_steps = args.max_sim_steps

    keep_context = args.keep_context
    reverse_betas = args.reverse_betas

    num_ini_tokens = args.num_ini_tokens
    reorder_weights = args.reorder_weights
    normalize_weights_str = args.normalize_weights_str
    compute_inf_normalization = args.compute_inf_normalization
    correlations_from_weights = args.correlations_from_weights  # 0 use gaussian corrs, 1 create from weight matrices, 2 uniform means, 3 segments
    gaussian_scale = args.gaussian_scale  # Only applicable if correlations_from_weights=0
    pe_mode = args.pe_mode
    num_segments_corrs = args.num_segments_corrs  # Only applicable if correlations_from_weights=3
    save_non_transient = args.save_non_transient
    save_not_plot = args.save_not_plot

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise ("You cannot discard more timesteps than you are simulating.")

    stats_to_save_plot = ["mo", "mo_se", "att"]
    # stats_to_save_plot = ["mo", "mo_se"]

    start = time.time()

    runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
           max_sim_steps, context_size, num_ini_tokens, seed_list, normalize_weights_str, reorder_weights,
           stats_to_save_plot, se_per_contribution, correlations_from_weights, num_segments_corrs, pe_mode,
           keep_context, reverse_betas, gaussian_scale, save_non_transient, compute_inf_normalization)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    ini_tokens_list = range(0, num_ini_tokens)
    plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed_list, normalize_weights_str,
            reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights, num_segments_corrs, pe_mode,
            se_per_contribution, keep_context, reverse_betas, gaussian_scale, save_non_transient,
            compute_inf_normalization)