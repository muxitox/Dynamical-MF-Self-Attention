import copy

import numpy as np
from models.HopfieldTransformerPE import HopfieldTransformer
from models.HopfieldTransformerMFInfNPE import HopfieldTransformerMFInfNPE

from bifurcation_diagrams import define_ini_token
from models.Embedding import Embedding
from plotting.plotting import plot_2_statistics

if __name__ == "__main__":

    ######
    # General variables
    ########
    seed = 101
    beta_att = 2.2
    beta_o = 500
    scaling_att = 100
    positional_embedding_size = 2
    scaling_o = 1
    normalize_weights_str_att = "N**2*np.sqrt(M)"
    normalize_weights_str_o = "N"
    max_sim_steps = 512
    min_saved_step = 100
    num_feat_patterns = 3
    context_size = 2 ** positional_embedding_size

    # Instantiate vocabulary
    semantic_embedding_size = 4
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    # We don't initialize the vocab as it's more efficient to work without a dict with the MF implementation
    vocab.initialize()

    # Define list of possible initial tokens
    np.random.seed(0)
    num_ini_tokens = 1
    ini_token_idx = 0
    ini_tokens_list = np.random.randint(2, size=(
    num_ini_tokens, semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1
    # Define wether to use a random initial token or to choose it from any of the patterns (Wo, Wv, Wq, Wk)
    ini_token_from_w = 0


    #################
    # STD Transformer
    #################

    reorder_weights = False

    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns, embedding_size, vocab, context_size,
                             max_sim_steps=max_sim_steps, min_saved_step=min_saved_step,
                             normalize_weights_str_att=normalize_weights_str_att,
                             normalize_weights_str_o=normalize_weights_str_o, reorder_weights=False, pe_mode=0,
                             weights_from_segments=True, scaling_o=scaling_o, scaling_att=scaling_att,
                             num_segments_corrs=3)


    x0 = copy.deepcopy(ini_tokens_list[ini_token_idx])
    define_ini_token(ini_token_from_w, HT, ini_token_idx, ini_tokens_list)

    HT.simulate(x0, max_steps=max_sim_steps)

    print(HT.x_list.shape)
    print(HT.x_list)

    # ################
    # # MF Transformer
    # ################
    #
    # np.random.seed(seed)
    #
    # # Instantiate vocabulary
    # tentative_semantic_embedding_size = 99
    # positional_embedding_size = 2
    # context_size = 2 ** positional_embedding_size
    # embedding_size = tentative_semantic_embedding_size + positional_embedding_size
    # vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size)
    # vocab.initialize_pos_encoder()
    #
    # # Create variables for the Hopfield Transformer (HT)
    # seed = 1
    # beta_list = [1.255, 1.26427, 1.266, 1.27, 1.28, 1.4]
    # beta_att = 2.2
    # num_feat_patterns = 3
    # num_transient_steps = 100000
    # saved_steps = 20000
    # max_sim_steps = num_transient_steps + saved_steps
    #
    # correlations_from_weights = 3
    # pe_mode = 0
    # se_per_contribution = 0.98
    # scaling_o = 1
    # scaling_att = 100
    #
    # normalize_weights_str_att = "N**2*np.sqrt(M)"
    # normalize_weights_str_o = "N"
    # compute_inf_normalization = True
    # save_not_plot = True
    # show_title = True
    #
    # # Create seed for reproducibility
    # np.random.seed(seed)
    #
    # HT = HopfieldTransformerMFInfNPE(beta, beta_att, num_feat_patterns=num_feat_patterns,
    #                                  positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
    #                                  context_size=context_size, max_sim_steps=max_sim_steps,
    #                                  min_saved_step=num_transient_steps,
    #                                  normalize_weights_str_att=normalize_weights_str_att,
    #                                  normalize_weights_str_o=normalize_weights_str_o,
    #                                  correlations_from_weights=correlations_from_weights,
    #                                  semantic_embedding_bitsize=tentative_semantic_embedding_size,
    #                                  epsilon_pe=se_per_contribution, pe_mode=pe_mode,
    #                                  compute_inf_normalization=compute_inf_normalization,
    #                                  scaling_o=scaling_o,
    #                                  scaling_att=scaling_att,
    #                                  N_normalization=9999)
    #
    #
    #
    # print("Simulating MF Transformer...")
    # HT.reset_data()
    # HT.simulate(x0, max_steps=max_sim_steps)
    # print("Done.")
    #
    # # print(HT.mf_statistics["mo"][0])
    # # print(HT.mf_statistics["mv"][0])
    # # print(HT.mf_statistics["att"][0])
    # # print(HT.mf_statistics["att"][1])
    #
    #
    # print("Same weights", np.array_equal(HTPEML.Wo, HT.Wo), np.array_equal(HTPEML.Wv, HT.Wv), np.array_equal(HTPEML.Wk, HT.Wk),
    #       np.array_equal(HTPEML.Wq, HT.Wq))
    #
    # # Plotting
    # print("Plotting statistics...")
    # num_plotting_steps = max_sim_steps
    # label_tag = ["finite", "inf"]
    # beta_str = r" $\beta$ =" + str(beta)
    #
    # label_tag = ["memory", "memoryless"]
    # for stat_name in HT.statistics_names:
    #
    #     if stat_name == "mk" or stat_name == "mv":
    #         statML = HTPEML.mf_statistics[stat_name][:, 0, :]
    #     else:
    #         statML = HTPEML.mf_statistics[stat_name]
    #
    #     plot_2_statistics(HT.mf_statistics[stat_name], statML, stat_name, num_feat_patterns,
    #                   num_plotting_steps, label_tag, additional_msg=beta_str)
    #
    #
    #
    #
    # print("Done.")
