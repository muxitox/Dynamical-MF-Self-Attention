import numpy as np
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from models.HopfieldTransformerPE import Embedding
from plotting.plotting import plot_save_statistics, plot_save_plane

if __name__ == "__main__":

    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 4
    context_size = 2 ** positional_embedding_size
    embedding_size = tentative_semantic_embedding_size + positional_embedding_size
    vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size)
    # vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    beta = 0.55
    beta_o = beta
    beta_att = beta

    num_feat_patterns = 3
    num_transient_steps_traj = 1070000 - 500  # 0 if we want to show the trajectory since the beginning
    num_transient_steps_plane = 1000000  # 0 if we want to show the trajectory since the beginning
    max_sim_steps = 1070000
    correlations_from_weights = 3
    pe_mode = 0
    se_per_contribution = tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)


    normalize_weights_str = "np.sqrt(N)*M"
    reorder_weights = False

    num_ini_tokens = 1
    # Select initial token with seed 0
    np.random.seed(0)
    ini_tokens_list = np.random.randint(2, size=(num_ini_tokens, tentative_semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    ini_token_idx = 0
    x0 = ini_tokens_list[ini_token_idx, :]


    # Interesting seeds: 18, 26
    # seed_list = range(30, 60)

    seed_list = [1]
    for seed in seed_list:

        # Create seed for reproducibility
        np.random.seed(seed)

        HT = HopfieldTransformerInfN(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                     positional_embedding_bitsize=positional_embedding_size, vocab=vocab, context_size=context_size,
                                     max_sim_steps=max_sim_steps, normalize_weights_str=normalize_weights_str,
                                     reorder_weights=reorder_weights, correlations_from_weights=correlations_from_weights,
                                     semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                     se_per_contribution=se_per_contribution, pe_mode = pe_mode)

        HT.reset_data()

        print(HT.even_corr_o_o)

        print("Simulating MF Transformer...")
        HT.simulate_mf(x0, max_steps=max_sim_steps)
        print("Done.")

        # Plotting
        print("Plotting statistics...")
        num_plotting_steps = max_sim_steps

        title = (f"MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed} "
                 f"BETA={beta} NUM_TRANSIENT={num_transient_steps_traj}")

        stats_to_show = ["mo_se"]
        for stat_name in stats_to_show:
            plot_save_statistics(HT.mf_statistics[stat_name][num_transient_steps_traj:,:], stat_name,
                                 num_feat_patterns, max_sim_steps-num_transient_steps_traj, title=title)
        print("Done.")


        # 3 feats

        stats_to_plot = ["mo_se", "mo_se"]
        stat_results_beta_list_0 = HT.mf_statistics[stats_to_plot[0]]
        stat_results_beta_list_1 = HT.mf_statistics[stats_to_plot[1]]
        feats_reorder = [1, 2, 0]
        plot_save_plane(stat_results_beta_list_0[num_transient_steps_plane:],
                        stat_results_beta_list_1[num_transient_steps_plane:, feats_reorder],
                        num_feat_patterns, max_sim_steps - num_transient_steps_plane, show_max_num_patterns=num_feat_patterns,
                        tag_names=stats_to_plot, beta=beta)

        # save_not_plot = False
        # save_path = ""
        #
        # stats_to_plot = ["mo", "mv"]
        # plot_save_plane(HT.mf_statistics[stats_to_plot[0]][num_transient_steps:], HT.mf_statistics[stats_to_plot[1]][num_transient_steps:], stat_name,
        #                 num_feat_patterns,
        #                 max_sim_steps - num_transient_steps,
        #                 show_max_num_patterns=num_feat_patterns,
        #                 save_not_plot=save_not_plot, tag_names=stats_to_plot, beta=beta)
        #
        # stats_to_plot = ["mk", "mq"]
        # plot_save_plane(HT.mf_statistics[stats_to_plot[0]][num_transient_steps:],
        #                 HT.mf_statistics[stats_to_plot[1]][num_transient_steps:], stat_name,
        #                 num_feat_patterns,
        #                 max_sim_steps - num_transient_steps,
        #                 show_max_num_patterns=num_feat_patterns,
        #                 save_not_plot=save_not_plot, tag_names=stats_to_plot, beta=beta)