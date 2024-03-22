import numpy as np
from models.HopfieldTransformerPE import Embedding, HopfieldTransformer



def compute_variances(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size,
           max_sim_steps, reorder_weights, seed_list):
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    # We don't initialize the vocab as it's more efficient to work without a dict with the MF implementation
    # vocab.initialize()

    for num_feat_patterns in num_feat_patterns_list:
        for seed in seed_list:
            np.random.seed(seed)

            # Initialize transformer weights and create variables for storing results
            HT = HopfieldTransformer(0, 0, num_feat_patterns=num_feat_patterns,
                                     embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps,
                                     reorder_weights=reorder_weights)

            Jatt = np.zeros((embedding_size, embedding_size))
            Jo = np.zeros((embedding_size, embedding_size))
            for a in range(num_feat_patterns):
                Jatt += HT.Wq.T @ HT.Wk
                Jo += HT.Wo.T @ HT.Wv

            Jatt_NMnorm = 1/(embedding_size) * 1/(num_feat_patterns) * Jatt
            Jo_NMnorm = 1/(embedding_size) * 1/(num_feat_patterns) * Jo

            Jatt_Nnorm = 1 / (embedding_size) * Jatt
            Jo_Nnorm = 1 / (embedding_size)  * Jo

            Jatt_sqrtNnorm = 1 / np.sqrt(embedding_size) * Jatt
            Jo_sqrtNnorm = 1 / np.sqrt(embedding_size) * Jo

            Jatt_sqrtMnorm = 1 / np.sqrt(num_feat_patterns) * Jatt
            Jo_sqrtMnorm = 1 / np.sqrt(num_feat_patterns) * Jo

            Jatt_sqrtNMnorm = 1 / np.sqrt(embedding_size) * 1 / np.sqrt(num_feat_patterns) * Jatt
            Jo_sqrtNMnorm = 1 / np.sqrt(embedding_size) * 1 / np.sqrt(num_feat_patterns) * Jo

            var_Jatt = np.var(Jatt)
            var_Jo = np.var(Jo)

            var_Jatt_Nnorm = np.var(Jatt_Nnorm)
            var_Jo_Nnorm = np.var(Jo_Nnorm)

            var_Jatt_sqrtNnorm = np.var(Jatt_sqrtNnorm)
            var_Jo_sqrtNnorm = np.var(Jo_sqrtNnorm)

            var_Jatt_sqrtMnorm = np.var(Jatt_sqrtMnorm)
            var_Jo_sqrtMnorm = np.var(Jo_sqrtMnorm)

            var_Jatt_NMnorm = np.var(Jatt_NMnorm)
            var_Jo_NMnorm = np.var(Jo_NMnorm)

            var_Jatt_sqrtNMnorm = np.var(Jatt_sqrtNMnorm)
            var_Jo_sqrtNMnorm = np.var(Jo_sqrtNMnorm)

            print("Seed", seed, "Num feat patterns(M)", num_feat_patterns, "Embedding_size(N)", embedding_size)
            print("Variances\t", "var_Jatt", "\t", "var_Jo", "\t", "1/N", "\t", "1/M", "\t", "1/(NM)")
            print("Variances\t", var_Jatt, "\t", var_Jo, "\t", 1/embedding_size, "\t", 1/num_feat_patterns, "\t", 1/(embedding_size*num_feat_patterns))
            print("Var 1/N norm\t", var_Jatt_Nnorm, "\t", var_Jo_Nnorm, "\t", 1/embedding_size, "\t", 1/num_feat_patterns, "\t", 1/(embedding_size*num_feat_patterns))
            print("Var 1/sqrt(N) norm\t", var_Jatt_sqrtNnorm, "\t", var_Jo_sqrtNnorm, "\t", 1/embedding_size, "\t", 1/num_feat_patterns, "\t", 1/(embedding_size*num_feat_patterns))
            print("Var 1/sqrt(M) norm\t", var_Jatt_sqrtMnorm, "\t", var_Jo_sqrtMnorm, "\t", 1/embedding_size, "\t", 1/num_feat_patterns, "\t", 1/(embedding_size*num_feat_patterns))
            print("Var 1/N 1/M norm\t", var_Jatt_NMnorm, "\t", var_Jo_NMnorm, "\t", 1/embedding_size, "\t", 1/num_feat_patterns, "\t", 1/(embedding_size*num_feat_patterns))
            print("Var 1/sqrt(N) 1/sqrt(M) norm\t", var_Jatt_sqrtNMnorm, "\t", var_Jo_sqrtNMnorm, "\t", 1/embedding_size, "\t", 1/num_feat_patterns, "\t", 1/(embedding_size*num_feat_patterns))
            print()

if __name__ == "__main__":
    # Instantiate vocabulary
    semantic_embedding_size = 10
    positional_embedding_size = 10

    # Create variables for the Hopfield Transformer (HT)
    seed_list = [3]
    num_feat_patterns_list = [10,100]
    num_transient_steps = 256
    max_sim_steps = 1024
    reorder_weights = False

    compute_variances(num_feat_patterns_list, semantic_embedding_size, positional_embedding_size,
                      max_sim_steps, reorder_weights, seed_list)
