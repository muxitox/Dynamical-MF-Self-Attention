import copy
import numpy as np
from scipy.special import softmax
from models.TransformerBase import TransformerBase


class HopfieldTransformerMFPE(TransformerBase):

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

        self.mv_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.mq_window = np.zeros(self.num_feat_patterns)
        self.mk_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.att_window = np.zeros(self.num_feat_patterns)

        # Create variables for saving the statistics of the mean-field model
        self.mf_statistics = {}
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, num_feat_patterns))

    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def reset_data(self):
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.max_sim_steps, self.num_feat_patterns))

    def reset_data_keep_context(self):
        mf_statistics_copy = {}
        for name_i in self.statistics_names:
            mf_statistics_copy[name_i] = copy.deepcopy(self.mf_statistics[name_i])

        self.reset_data()

        for name_i in self.statistics_names:
            self.mf_statistics[name_i][:self.context_size, :] = mf_statistics_copy[name_i][-self.context_size:, :]

    def define_total_normalization_o(self):
        total_normalization = self.embedding_size / self.normalizing_constant_o * self.scaling_o

        return total_normalization

    def define_total_normalization_att(self):
        total_normalization = self.embedding_size ** 2 / self.normalizing_constant_att * self.scaling_att

        return total_normalization

    def qk_f_mf(self, t, tau):
        mqk = self.mf_statistics["mq"][t] @ self.mf_statistics["mk"][tau]
        return self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk

    def attention_unoptimized(self, t):

        idx_ctx_start = max(0, t - self.context_size + 1)
        effective_context_size = min(self.context_size, t + 1)

        key_prob = np.zeros(effective_context_size)
        for tau in range(0, effective_context_size):
            key_prob[tau] = self.qk_f_mf(t, idx_ctx_start + tau)
        key_prob = softmax(key_prob)

        att_t = self.embedding_size * (self.mf_statistics["mv"][idx_ctx_start:t + 1].T @ key_prob)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        self.mf_statistics["att"][t] = att_t

        return att_t

    def attention_mf_optimized_2(self, t):

        effective_context_size = min(self.context_size, t + 1)

        mqk = np.einsum('b,tb -> t', self.mq_window, self.mk_window[:effective_context_size],
                        optimize=True)

        key_prob_unnorm = self.beta_att * self.total_normalization_att * mqk

        key_prob = softmax(key_prob_unnorm)
        self.att_window = self.mv_window[:effective_context_size].T @ key_prob

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        if t >= self.min_saved_step:
            self.mf_statistics["att"][t - self.min_saved_step] = copy.deepcopy(self.att_window)

    def attention(self, t):

        idx_ctx_start = max(0, t - self.context_size + 1)

        mqk = np.einsum('b,tb -> t', self.mf_statistics["mq"][t], self.mf_statistics["mk"][idx_ctx_start:t + 1],
                        optimize=True)

        key_prob_unnorm = self.beta_att * self.total_normalization_att * mqk
        key_prob = softmax(key_prob_unnorm)
        att_t = (self.mf_statistics["mv"][idx_ctx_start:t + 1].T @ key_prob)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        if t >= self.min_saved_step:
            self.mf_statistics["att"][t - self.min_saved_step] = att_t

        return att_t

    def compute_means_from_data(self, x0, t):
        self.mf_statistics["mo"][t] = x0 @ self.Wo.T / self.embedding_size
        self.mf_statistics["mo_se"][t] = x0[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.se_bit_size
        self.mf_statistics["mv"][t] = x0 @ self.Wv.T / self.embedding_size
        self.mf_statistics["mq"][t] = x0 @ self.Wq.T / self.embedding_size
        self.mf_statistics["mk"][t] = x0 @ self.Wk.T / self.embedding_size

    def compute_mf(self, t, att):

        # Compute the mean of every (semantic) spin i at time t
        att_Wo_i = np.tanh(self.beta_o * self.total_normalization_o *
                           np.einsum('b,bi -> i', att, self.Wo[:, :self.se_bit_size], optimize=True))

        # Concatenate semantic information with positional encoding
        pos_vec = self.vocab.encode_pos(t % self.context_size)
        att_Wo_i = np.concatenate((att_Wo_i, pos_vec))

        # Compute only semantic information
        unnorm_mo_se = np.einsum('bi,i ->b', self.Wo[:, :self.se_bit_size],
                                 att_Wo_i[:self.se_bit_size], optimize=True)

        # Normalize and save it. m^o is computed differently without PE since we want to analyze it separately.
        self.mf_statistics["mo_se"][t] = unnorm_mo_se / self.se_bit_size

        # Compute position information for m^o
        pe_contribution_o = np.einsum('bi,i ->b', self.Wo[:, self.se_bit_size:], pos_vec, optimize=True)

        # Compute mean-fields
        self.mf_statistics["mo"][t] = (unnorm_mo_se + pe_contribution_o) / self.embedding_size
        self.mf_statistics["mv"][t] = np.einsum('bi,i ->b', self.Wv, att_Wo_i, optimize=True) / self.embedding_size
        self.mf_statistics["mq"][t] = np.einsum('bi,i ->b', self.Wq, att_Wo_i, optimize=True) / self.embedding_size
        self.mf_statistics["mk"][t] = np.einsum('bi,i ->b', self.Wk, att_Wo_i, optimize=True) / self.embedding_size

        # # Loopy implementation for testing
        # mo_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         mo_t[b] += self.Wo[b, i] * np.tanh(self.beta_o * (1 / self.normalizing_constant) *  self.Wo[:, i] @ att)
        # mo_t /= self.embedding_size
        # print(np.allclose(self.mo[t], mo_t))

    def simulate_mf_from_context(self, max_steps):
        # In order for this method to work properly, a simulate_mf() method has had to be run previously at least for
        # self.context_size steps

        # Initialize attention to the last computed attention
        att = self.mf_statistics["att"][self.context_size - 1, :]

        # We initialize the model at the end of the previous
        ini_t = self.context_size
        for t in range(ini_t, max_steps):
            self.compute_mf(t, att)
            att = self.attention(t)

    def simulate(self, x0, max_steps):

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(x0, t=0)
        att = self.attention(t=0)

        for t in range(1, max_steps):
            self.compute_mf(t, att)
            att = self.attention(t)
