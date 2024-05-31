import copy
import numpy as np
from scipy.special import softmax
from models.TransformerBase import TransformerBase


class HopfieldTransformer(TransformerBase):

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, context_size, max_sim_steps=512,
                 min_saved_step=0,
                 normalize_weights_str_att="N**2", normalize_weights_str_o="N", reorder_weights=False, pe_mode=0,
                 weights_from_segments=False, scaling_o=1, scaling_att=1, num_segments_corrs=3):

        self.context_size = context_size
        self.context_index = 0

        N = embedding_size
        TransformerBase.__init__(self, beta_o, beta_att, num_feat_patterns, vocab.pe_bit_size, vocab,
                                 context_size, N,
                                 max_sim_steps=max_sim_steps, min_saved_step=min_saved_step,
                                 normalize_weights_str_att=normalize_weights_str_att,
                                 normalize_weights_str_o=normalize_weights_str_o,
                                 reorder_weights=reorder_weights, pe_mode=pe_mode,
                                 semantic_embedding_bitsize=vocab.se_bit_size,
                                 scaling_o=scaling_o, scaling_att=scaling_att)

        self.total_normalization_o = self.define_total_normalization_o()
        self.total_normalization_att = self.define_total_normalization_att()

        self.create_W_matrices_finite_model(weights_from_segments, num_segments_corrs)
        self.define_pair_correlations_from_weights()
        if num_feat_patterns >= 3:
            self.define_quad_correlations_from_weights()

        show_decoded_tokens = False  # Hardcoded, compute representation of the tokens codified in W
        if show_decoded_tokens:
            self.decoded_tokens = np.zeros(len(self.W))
            for a in range(0, len(self.W)):
                self.decoded_tokens[a] = vocab.decode(self.W[a])

        # List to save selected tokens in the standard model execution
        self.x_list = np.zeros((self.num_saved_steps, embedding_size))
        # Window for the x values
        self.x_window = np.zeros((self.context_size, embedding_size))

        # Create variables for saving the statistics of the standard model corresponding to the mean-field
        self.mf_statistics = {}
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, num_feat_patterns))

    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def reset_data(self):
        self.x_list = np.zeros((self.max_sim_steps, self.embedding_size))

        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.max_sim_steps, self.num_feat_patterns))

    def reset_data_keep_context(self):
        x_list_copy = copy.deepcopy(self.x_list)

        mf_statistics_copy = {}
        for name_i in self.statistics_names:
            mf_statistics_copy[name_i] = copy.deepcopy(self.mf_statistics[name_i])

        self.reset_data()

        self.x_list[:self.context_size, :] = x_list_copy[-self.context_size:, :]
        for name_i in self.statistics_names:
            self.mf_statistics[name_i][:self.context_size, :] = mf_statistics_copy[name_i][-self.context_size:, :]

    def define_total_normalization_o(self):
        total_normalization = self.embedding_size / self.normalizing_constant_o * self.scaling_o

        return total_normalization

    def define_total_normalization_att(self):
        total_normalization = self.embedding_size ** 2 / self.normalizing_constant_att * self.scaling_att

        return total_normalization

    def qk_f(self, t, tau):

        q = self.x_window[self.context_index] @ self.Wq.T  # Query representation
        k = self.Wk @ self.x_window[tau]  # Key representation

        # Save the statistics for comparison with the MF approximation
        if t == tau and t >= self.min_saved_step:
            self.mf_statistics["mq"][t - self.min_saved_step] = q / self.embedding_size
            self.mf_statistics["mk"][t - self.min_saved_step] = k / self.embedding_size

        qk = q @ k

        # res = np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * qk)
        res = np.exp(self.beta_att * (1 / self.normalizing_constant_att) * qk)


        # # Loopy implementation for testing
        # qk_accum = 0
        # for a in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         for j in range(0, self.embedding_size):
        #             qk_accum += self.x_list[t,i] * self.Wq[a, i] * self.Wk[a, j] * self.x_list[tau,j]
        #
        # res2 = self.beta_att * (1 / self.normalizing_constant) * qk_accum
        # print(np.allclose(res, res2))

        return res

    def attention(self, t):
        effective_context_size = min(self.context_size, t + 1)

        key_prob = np.zeros(effective_context_size)
        for tau in range(0, effective_context_size):
            key_prob[tau] = self.qk_f(t, tau)
        key_prob /= np.sum(key_prob)

        v = self.x_list[:effective_context_size] @ self.Wv.T  # Value representation
        att_t = key_prob @ v / self.embedding_size # Normalize A by N

        # Save for stats comparison
        if t >= self.min_saved_step:
            self.mf_statistics["mv"][t - self.min_saved_step] = v[-1] / self.embedding_size

        # # Loopy implementation for testing
        # att_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         for tau in range(0, t + 1):
        #             att_t[b] += self.Wv[b, i] * self.x_list[tau,i] * key_prob[tau]

        self.mf_statistics["att"][t - self.min_saved_step] = att_t

        return att_t

    def simulate(self, x0, max_steps, verbose=False):

        self.x_list[0, :] = x0

        # Save for comparison with MF

        if 0 == self.min_saved_step:
            self.mf_statistics["mo"][0] = x0 @ self.Wo.T / self.embedding_size
            self.mf_statistics["mo_se"][0] = x0[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size

        selected_tokens = []

        for t in range(0, max_steps):

            self.context_index = t % self.context_size

            att = self.attention(t)

            if t < max_steps - 1:  # We'll compute att once more for computing statistics


                # We project all possible tokens in the vocabulary through Wo
                o = self.vocab.idx2word @ self.Wo.T

                # We multiply by the attention score
                prob_unnormalized = self.beta_o * (1 / self.normalizing_constant_o) * o @ att

                prob_normalized = softmax(prob_unnormalized)

                if verbose == True:
                    print("Num tokens with max probability in time", t, ":",
                          np.sum(np.isclose(prob_normalized, max(prob_normalized))), "/", self.vocab.vocab_size)

                # Convert the above result into a probability and get the idx of the most probable token
                sample = True
                if sample:
                    new_x_idx = np.random.choice(range(len(prob_normalized)), p=prob_normalized)
                else:
                    new_x_idx = np.argmax(prob_normalized)

                # Encode token and add it to the list
                # new_x = self.vocab.encode(new_x_idx)
                self.x_list[t + 1, :] = self.vocab.encode_w_pos(new_x_idx, (t + 1) % self.context_size)

                # Save for comparison with MF
                if t >= self.min_saved_step:
                    self.mf_statistics["mo"][t + 1 - self.min_saved_step] = self.x_list[t + 1, :] @ self.Wo.T / self.embedding_size
                    self.mf_statistics["mo_se"][t + 1 - self.min_saved_step] = self.x_list[t + 1, :][:self.se_bit_size] @ self.Wo[:,
                                                                        :self.se_bit_size].T / self.embedding_size

                selected_tokens.append(new_x_idx)

        return selected_tokens
