import copy

import numpy as np
from models.HopfieldTransformerPE import HopfieldTransformer
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from models.HopfieldTransformerPE import Embedding
from plotting.plotting import plot_2_statistics

if __name__ == "__main__":

    # Instantiate vocabulary
    semantic_embedding_size = 100
    positional_embedding_size = 4
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    # We don't initialize the vocab as it's more efficient to work without a dict with the MF implementation
    # vocab.initialize()

    np.random.seed(0)
    num_ini_tokens = 5
    ini_token_idx = 2

    ini_tokens_list = np.random.randint(2, size=(num_ini_tokens, semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1
    x0 = copy.deepcopy(ini_tokens_list[ini_token_idx])

    # vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    seed = 4
    beta = 3
    beta_o = beta
    beta_att = beta

    num_feat_patterns = 1
    max_sim_steps = 5
    context_size = 2 ** positional_embedding_size

    normalize_weights_str = "np.sqrt(N*M)"
    reorder_weights = False

    # Create seed for reproducibility
    np.random.seed(seed)

    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                             embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps, context_size=context_size,
                             normalize_weights_str=normalize_weights_str, reorder_weights=reorder_weights)

    num_runs = 1

    print("Simulating MF Transformer...")
    HT.reset_data()
    HT.simulate_mf(x0, max_steps=max_sim_steps)
    print("Done.")

    np.random.seed(seed)
    correlation_from_weights = 1
    normalize_weights_str = "np.sqrt(N*M)"
    se_per_contribution = semantic_embedding_size / (semantic_embedding_size + positional_embedding_size)
    HTInf = HopfieldTransformerInfN(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                 positional_embedding_bitsize=positional_embedding_size, context_size=context_size,
                                 max_sim_steps=max_sim_steps, normalize_weights_str=normalize_weights_str,
                                 reorder_weights=reorder_weights, correlations_from_weights=correlation_from_weights,
                                 semantic_embedding_bitsize=semantic_embedding_size,
                                 se_per_contribution=se_per_contribution)


    print("Simulating Inf MF Transformer...")
    HTInf.reset_data()
    HTInf.simulate_mf(x0, max_steps=max_sim_steps)
    print("Done.")

    print("Same weights", np.array_equal(HTInf.Wo, HT.Wo), np.array_equal(HTInf.Wv, HT.Wv), np.array_equal(HTInf.Wk, HT.Wk),
          np.array_equal(HTInf.Wq, HT.Wq))

    print(HT.mf_statistics["mo"])
    print(HTInf.mf_statistics["mo"])

    # Plotting
    print("Plotting statistics...")
    num_plotting_steps = max_sim_steps
    label_tag = ["finite", "inf"]
    beta_str = r" $\beta$ =" + str(beta)
    for stat_name in HT.statistics_names:
        plot_2_statistics(HT.mf_statistics[stat_name], HTInf.mf_statistics[stat_name], stat_name, num_feat_patterns,
                      num_plotting_steps, label_tag, additional_msg=beta_str)
    print("Done.")
