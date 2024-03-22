import copy
import numpy as np
from scipy.special import softmax
from utils import bitfield, bool2int

class HopfieldTransformerInfN:

    def __init__(self, beta_o, beta_att, num_feat_patterns, positional_embedding_bitsize, context_size, max_sim_steps=512,
                 normalize_weights_str="1", reorder_weights=False, correlations_from_weights=True, semantic_embedding_bitsize=0,
                 se_per_contribution=0.95):

        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = semantic_embedding_bitsize
        self.pe_bit_size = positional_embedding_bitsize
        self.se_per_contribution = se_per_contribution

        self.embedding_size = semantic_embedding_bitsize + positional_embedding_bitsize

        self.context_size = context_size

        N = self.embedding_size
        M = num_feat_patterns

        # Dynamically compute the normalize_weights_str string
        try:
            exec_str = f"self.normalizing_constant = {normalize_weights_str}"
            exec(exec_str)
        except:
            print("The exec_str string is not well defined")
            raise


        self.W = np.random.randint(2, size=(num_feat_patterns, self.embedding_size)) * 2 - 1

        if reorder_weights:
            self.Wo = np.copy(self.W)
            np.random.shuffle(self.Wo)
            self.Wv = np.copy(self.W)
            np.random.shuffle(self.Wv)
            # self.Wv = np.roll(self.Wo, 1, 1)
            self.Wq = np.copy(self.W)
            np.random.shuffle(self.Wq)
            self.Wk = np.copy(self.W)
            np.random.shuffle(self.Wk)
            # self.Wk = self.Wq

        else:
            self.Wo = np.random.randint(2, size=(num_feat_patterns, self.embedding_size)) * 2 - 1
            self.Wv = np.random.randint(2, size=(num_feat_patterns, self.embedding_size)) * 2 - 1
            self.Wq = np.random.randint(2, size=(num_feat_patterns, self.embedding_size)) * 2 - 1
            self.Wk = np.random.randint(2, size=(num_feat_patterns, self.embedding_size)) * 2 - 1

            self.W = self.Wo


        if correlations_from_weights == 1:  # Create matrices and compute correlations from them

            # Create correlations from matrices for comparison
            if num_feat_patterns < 1 or num_feat_patterns > 3:
                raise "The number of patterns is neither 1, 2 or 3"

            self.pair_corr_o_o = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_v = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_k = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_q = np.zeros((num_feat_patterns, num_feat_patterns))

            for b in range(0, num_feat_patterns):
                for a in range(0, num_feat_patterns):
                    for i in range(0, semantic_embedding_bitsize):
                        self.pair_corr_o_o[a, b] += self.Wo[a, i] * self.Wo[b, i]
                        self.pair_corr_o_v[a, b] += self.Wo[a, i] * self.Wv[b, i]
                        self.pair_corr_o_k[a, b] += self.Wo[a, i] * self.Wk[b, i]
                        self.pair_corr_o_q[a, b] += self.Wo[a, i] * self.Wq[b, i]

            self.pair_corr_o_o /= semantic_embedding_bitsize
            self.pair_corr_o_v /= semantic_embedding_bitsize
            self.pair_corr_o_q /= semantic_embedding_bitsize
            self.pair_corr_o_k /= semantic_embedding_bitsize

            if num_feat_patterns == 3:
                self.quad_corr_o_o = np.zeros(num_feat_patterns)
                self.quad_corr_o_v = np.zeros(num_feat_patterns)
                self.quad_corr_o_k = np.zeros(num_feat_patterns)
                self.quad_corr_o_q = np.zeros(num_feat_patterns)

                for b in range(0, num_feat_patterns):
                    for i in range(0, semantic_embedding_bitsize):
                        Wo_corr = self.Wo[0, i] * self.Wo[1, i] * self.Wo[2, i]
                        self.quad_corr_o_o[b] += Wo_corr * self.Wo[b, i]
                        self.quad_corr_o_v[b] += Wo_corr * self.Wv[b, i]
                        self.quad_corr_o_q[b] += Wo_corr * self.Wq[b, i]
                        self.quad_corr_o_k[b] += Wo_corr * self.Wk[b, i]

                self.quad_corr_o_o /= semantic_embedding_bitsize
                self.quad_corr_o_v /= semantic_embedding_bitsize
                self.quad_corr_o_q /= semantic_embedding_bitsize
                self.quad_corr_o_v /= semantic_embedding_bitsize

        elif correlations_from_weights == 0:  #  Normal weights and 4 correlations come from individual corrs
            sc = 0.5
            # Create random correlations
            self.pair_corr_o_o = np.random.normal(0, sc, (num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_v = np.random.normal(0, sc, (num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_k = np.random.normal(0, sc, (num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_q = np.random.normal(0, sc, (num_feat_patterns, num_feat_patterns))

            # Set autocorrelations to 0
            np.fill_diagonal(self.pair_corr_o_o, 1)

            self.pair_corr_o_o = np.clip(self.pair_corr_o_o, -1, 1)
            self.pair_corr_o_v = np.clip(self.pair_corr_o_v, -1, 1)
            self.pair_corr_o_k = np.clip(self.pair_corr_o_k, -1, 1)
            self.pair_corr_o_q = np.clip(self.pair_corr_o_q, -1, 1)

            if num_feat_patterns == 3:
                # self.three_corr_o_o = np.random.normal(0, sc, num_feat_patterns)
                # self.three_corr_o_v = np.random.normal(0, sc, num_feat_patterns)
                # self.three_corr_o_k = np.random.normal(0, sc, num_feat_patterns)
                # self.three_corr_o_q = np.random.normal(0, sc, num_feat_patterns)
                #
                # self.three_corr_o_o = np.clip(self.three_corr_o_o, -1, 1)
                # self.three_corr_o_v = np.clip(self.three_corr_o_v, -1, 1)
                # self.three_corr_o_k = np.clip(self.three_corr_o_k, -1, 1)
                # self.three_corr_o_q = np.clip(self.three_corr_o_q, -1, 1)

                self.quad_corr_o_o = np.prod(self.pair_corr_o_o, axis=0)
                self.quad_corr_o_v = np.prod(self.pair_corr_o_v, axis=0)
                self.quad_corr_o_k = np.prod(self.pair_corr_o_k, axis=0)
                self.quad_corr_o_q = np.prod(self.pair_corr_o_q, axis=0)

        elif correlations_from_weights == 2:  # Correlations from uniform means

            # Create initial means that will later define the correlations
            self.corr_mo = np.random.rand(num_feat_patterns)
            self.corr_mv = np.random.rand(num_feat_patterns)
            self.corr_mq = np.random.rand(num_feat_patterns)
            self.corr_mk = np.random.rand(num_feat_patterns)

            self.pair_corr_o_o = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_v = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_k = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_q = np.zeros((num_feat_patterns, num_feat_patterns))


            for b in range(0, num_feat_patterns):
                for a in range(0, num_feat_patterns):
                    self.pair_corr_o_o[a, b] += self.corr_mo[a] * self.corr_mo[b]
                    self.pair_corr_o_v[a, b] += self.corr_mo[a] * self.corr_mv[b]
                    self.pair_corr_o_q[a, b] += self.corr_mo[a] * self.corr_mq[b]
                    self.pair_corr_o_k[a, b] += self.corr_mo[a] * self.corr_mk[b]

            np.fill_diagonal(self.pair_corr_o_o, 1)

            if num_feat_patterns == 3:
                self.quad_corr_o_o = np.zeros(num_feat_patterns)
                self.quad_corr_o_v = np.zeros(num_feat_patterns)
                self.quad_corr_o_k = np.zeros(num_feat_patterns)
                self.quad_corr_o_q = np.zeros(num_feat_patterns)

                for b in range(0, num_feat_patterns):
                    Wo_corr = self.corr_mo[0] * self.corr_mo[1] * self.corr_mo[2]
                    Wo_corr_mo = self.corr_mo[(b+1) % num_feat_patterns] * self.corr_mo[(b+2) % num_feat_patterns]
                    self.quad_corr_o_o[b] += Wo_corr_mo
                    self.quad_corr_o_v[b] += Wo_corr * self.corr_mv[b]
                    self.quad_corr_o_k[b] += Wo_corr * self.corr_mk[b]
                    self.quad_corr_o_q[b] += Wo_corr * self.corr_mq[b]

        self.even_corr_o_o = self.pair_corr_o_o
        self.even_corr_o_v = self.pair_corr_o_v
        self.even_corr_o_k = self.pair_corr_o_k
        self.even_corr_o_q = self.pair_corr_o_q

        if num_feat_patterns == 1:
            self.sign_matrix = np.ones((1,1))  # Create empty dimension in order to allow matrix indexing later
        elif num_feat_patterns == 2:
            self.sign_matrix = np.array([[1, 1], [1, -1]])
        elif num_feat_patterns == 3:
            self.sign_matrix = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])

            self.even_corr_o_o = np.vstack((self.pair_corr_o_o, self.quad_corr_o_o))
            self.even_corr_o_v = np.vstack((self.pair_corr_o_v, self.quad_corr_o_v))
            self.even_corr_o_k = np.vstack((self.pair_corr_o_k, self.quad_corr_o_k))
            self.even_corr_o_q = np.vstack((self.pair_corr_o_q, self.quad_corr_o_q))

        self.num_feat_patterns = num_feat_patterns
        self.max_sim_steps = max_sim_steps

        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        # Create variables for saving the statistics of the mean-field model
        self.mf_statistics = {}
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((max_sim_steps, num_feat_patterns))


    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def reset_data(self):
        self.x_list = np.zeros((self.max_sim_steps, self.embedding_size))

        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.max_sim_steps, self.num_feat_patterns))


    def reset_data_keep_context(self):

        mf_statistics_copy = {}
        for name_i in self.statistics_names:
            mf_statistics_copy[name_i] = copy.deepcopy(self.mf_statistics[name_i])

        self.reset_data()

        for name_i in self.statistics_names:
            self.mf_statistics[name_i][:self.context_size, :] = mf_statistics_copy[name_i][-self.context_size:, :]

    def qk_f_mf(self, t, tau):

        mqk = self.mf_statistics["mq"][t] @ self.mf_statistics["mk"][tau]
        return self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk

    def attention_mf(self, t):

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

    def attention_mf_optimized(self, t):

        idx_ctx_start = max(0, t - self.context_size + 1)

        mqk = np.einsum('b,tb -> t', self.mf_statistics["mq"][t], self.mf_statistics["mk"][idx_ctx_start:t + 1], optimize=True)

        # key_prob = self.beta_att * self.embedding_size ** 2 / np.sqrt(self.num_feat_patterns) * mqk
        key_prob = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk
        # key_prob = self.beta_att * self.embedding_size ** 2 * mqk

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

    def compute_means_from_data(self, x0, t):
        self.mf_statistics["mo"][t] = x0 @ self.Wo.T / self.embedding_size
        self.mf_statistics["mo_se"][t] = x0[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.se_bit_size
        self.mf_statistics["mv"][t] = x0 @ self.Wv.T / self.embedding_size
        self.mf_statistics["mq"][t] = x0 @ self.Wq.T / self.embedding_size
        self.mf_statistics["mk"][t] = x0 @ self.Wk.T / self.embedding_size


    def compute_mf_optimized(self, t, att):

        pe_contribution_o = np.einsum('bi,i ->b', self.Wo[:, self.se_bit_size:],
                                    bitfield(t % self.context_size, self.pe_bit_size) * 2 - 1,
                                    optimize=True) / self.pe_bit_size

        pe_contribution_v = np.einsum('bi,i ->b', self.Wv[:, self.se_bit_size:],
                                    bitfield(t % self.context_size, self.pe_bit_size) * 2 - 1,
                                    optimize=True) / self.pe_bit_size

        pe_contribution_q = np.einsum('bi,i ->b', self.Wq[:, self.se_bit_size:],
                                    bitfield(t % self.context_size, self.pe_bit_size) * 2 - 1,
                                    optimize=True) / self.pe_bit_size

        pe_contribution_k = np.einsum('bi,i ->b', self.Wk[:, self.se_bit_size:],
                                    bitfield(t % self.context_size, self.pe_bit_size) * 2 - 1,
                                    optimize=True) / self.pe_bit_size


        tanh_j_sings = np.tanh(
            self.beta_o * (1 / self.normalizing_constant) * np.einsum("jb,b->j",
                                                                      self.sign_matrix[:, :self.num_feat_patterns],
                                                                      att))

        corr_signed_o = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_o)
        corr_signed_v = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_v)
        corr_signed_q = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_q)
        corr_signed_k = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_k)

        self.mf_statistics["mo_se"][t] = np.einsum("jb,j->b", corr_signed_o, tanh_j_sings) / 2 ** (
                    self.num_feat_patterns - 1)
        self.mf_statistics["mo"][t] = self.se_per_contribution * self.mf_statistics["mo_se"][t] + (1 - self.se_per_contribution) * pe_contribution_o

        mv_se = self.se_per_contribution * np.einsum("jb,j->b", corr_signed_v, tanh_j_sings) / 2 ** (
                    self.num_feat_patterns - 1)
        self.mf_statistics["mv"][t] = mv_se + (1 - self.se_per_contribution) * pe_contribution_v

        mq_se = self.se_per_contribution * np.einsum("jb,j->b", corr_signed_q, tanh_j_sings) / 2 ** (
                    self.num_feat_patterns - 1)
        self.mf_statistics["mq"][t] = mq_se + (1 - self.se_per_contribution) * pe_contribution_q

        mk_se = self.se_per_contribution * np.einsum("jb,j->b", corr_signed_k, tanh_j_sings) / 2 ** (
                    self.num_feat_patterns - 1)
        self.mf_statistics["mk"][t] = mk_se + (1 - self.se_per_contribution) * pe_contribution_k

    def compute_mf(self, t, att):

        pe_contribution = np.einsum('bi,i ->b', self.Wo[:, self.se_bit_size:],
                                    bitfield(t % self.context_size, self.pe_bit_size) * 2 - 1,
                                    optimize=True) / self.embedding_size

        if self.num_feat_patterns == 1:
            tanh_b = np.tanh(self.beta_o * (1 / self.normalizing_constant) * att[0])

            self.mf_statistics["mo_se"][t] = self.se_per_contribution * self.pair_corr_o_o * tanh_b
            self.mf_statistics["mo"][t] = self.mf_statistics["mo_se"][t] + (1 - self.se_per_contribution) * pe_contribution
            self.mf_statistics["mv"][t] = (self.se_per_contribution * self.pair_corr_o_v * tanh_b +
                                           (1 - self.se_per_contribution) * pe_contribution)
            self.mf_statistics["mq"][t] = (self.se_per_contribution * self.pair_corr_o_q * tanh_b +
                                           (1 - self.se_per_contribution) * pe_contribution)
            self.mf_statistics["mk"][t] = (self.se_per_contribution * self.pair_corr_o_k * tanh_b +
                                           (1 - self.se_per_contribution) * pe_contribution)

        elif self.num_feat_patterns == 2:

            tanh_b_plus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] + att[1]))
            tanh_b_minus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] - att[1]))

            for b in range(0, self.num_feat_patterns):

                self.mf_statistics["mo_se"][t, b] = (self.se_per_contribution *
                            ((self.pair_corr_o_o[0,b] + self.pair_corr_o_o[1,b]) * tanh_b_plus
                             + (self.pair_corr_o_o[0,b] - self.pair_corr_o_o[1,b]) * tanh_b_minus) / 2 )

                self.mf_statistics["mo"][t, b] = (self.mf_statistics["mo_se"][t,b] +
                                                  (1 - self.se_per_contribution) * pe_contribution[b])

                self.mf_statistics["mv"][t, b] = (self.se_per_contribution *
                            ((self.pair_corr_o_v[0,b] + self.pair_corr_o_v[1,b]) * tanh_b_plus +
                             (self.pair_corr_o_v[0,b] - self.pair_corr_o_v[1,b]) * tanh_b_minus ) / 2
                                + (1 - self.se_per_contribution) * pe_contribution[b])

                self.mf_statistics["mq"][t, b] = (self.se_per_contribution *
                            ((self.pair_corr_o_q[0, b] + self.pair_corr_o_q[1, b]) * tanh_b_plus +
                             (self.pair_corr_o_q[0, b] - self.pair_corr_o_q[1, b]) * tanh_b_minus ) / 2
                                + (1 - self.se_per_contribution) * pe_contribution[b])

                self.mf_statistics["mk"][t, b] = (self.se_per_contribution *
                            ((self.pair_corr_o_k[0, b] + self.pair_corr_o_k[1, b]) * tanh_b_plus +
                             (self.pair_corr_o_k[0, b] - self.pair_corr_o_k[1, b]) * tanh_b_minus ) / 2
                              + (1 - self.se_per_contribution) * pe_contribution[b])

        else:
            tanh_b_plus_plus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] + att[1] + att[2]))
            tanh_b_plus_minus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] + att[1] - att[2]))
            tanh_b_minus_plus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] - att[1] + att[2]))
            tanh_b_minus_minus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] - att[1] - att[2]))

            for b in range(0, self.num_feat_patterns):
                self.mf_statistics["mo_se"][t, b] = (self.se_per_contribution *
                                                     ((self.pair_corr_o_o[0, b] + self.pair_corr_o_o[1, b] + self.pair_corr_o_o[2, b] + self.quad_corr_o_o[b]) * tanh_b_plus_plus
                                                      + (self.pair_corr_o_o[0, b] + self.pair_corr_o_o[1, b] - self.pair_corr_o_o[2, b] - self.quad_corr_o_o[b]) * tanh_b_plus_minus
                                                      + (self.pair_corr_o_o[0, b] - self.pair_corr_o_o[1, b] + self.pair_corr_o_o[2, b] - self.quad_corr_o_o[b]) * tanh_b_minus_plus
                                                      + (self.pair_corr_o_o[0, b] - self.pair_corr_o_o[1, b] - self.pair_corr_o_o[2, b] +
                                                         self.quad_corr_o_o[b]) * tanh_b_minus_minus) / 4)

                self.mf_statistics["mo"][t, b] = (self.mf_statistics["mo_se"][t,b] +
                                                  (1 - self.se_per_contribution) * pe_contribution[b])

                self.mf_statistics["mv"][t, b] = (self.se_per_contribution *
                                                  ((self.pair_corr_o_v[0, b] + self.pair_corr_o_v[1, b] + self.pair_corr_o_v[2, b]
                                                    + self.quad_corr_o_v[b]) * tanh_b_plus_plus
                                                   + (self.pair_corr_o_v[0, b] + self.pair_corr_o_v[1, b] - self.pair_corr_o_v[2, b]
                                                      - self.quad_corr_o_v[b]) * tanh_b_plus_minus
                                                   + (self.pair_corr_o_v[0, b] - self.pair_corr_o_v[1, b] + self.pair_corr_o_v[2, b]
                                                      - self.quad_corr_o_v[b]) * tanh_b_minus_plus
                                                   + (self.pair_corr_o_v[0, b] - self.pair_corr_o_v[1, b] - self.pair_corr_o_v[2, b]
                                                      + self.quad_corr_o_v[b]) * tanh_b_minus_minus) / 4 +
                                                  (1 - self.se_per_contribution) * pe_contribution[b])

                self.mf_statistics["mq"][t, b] = (self.se_per_contribution *
                                                  ((self.pair_corr_o_q[0, b] + self.pair_corr_o_q[1, b] + self.pair_corr_o_q[2, b]
                                                    + self.quad_corr_o_q[b]) * tanh_b_plus_plus
                                                   + (self.pair_corr_o_q[0, b] + self.pair_corr_o_q[1, b] - self.pair_corr_o_q[2, b]
                                                      - self.quad_corr_o_q[b]) * tanh_b_plus_minus
                                                   + (self.pair_corr_o_q[0, b] - self.pair_corr_o_q[1, b] + self.pair_corr_o_q[2, b]
                                                      - self.quad_corr_o_q[b]) * tanh_b_minus_plus
                                                   + (self.pair_corr_o_q[0, b] - self.pair_corr_o_q[1, b] - self.pair_corr_o_q[2, b]
                                                      + self.quad_corr_o_q[b]) * tanh_b_minus_minus) / 4 +
                                                  (1 - self.se_per_contribution) * pe_contribution[b])

                self.mf_statistics["mk"][t, b] = (self.se_per_contribution *
                                                  ((self.pair_corr_o_k[0, b] + self.pair_corr_o_k[1, b] + self.pair_corr_o_k[2, b]
                                                    + self.quad_corr_o_k[b]) * tanh_b_plus_plus
                                                   + (self.pair_corr_o_k[0, b] + self.pair_corr_o_k[1, b] - self.pair_corr_o_k[2, b]
                                                      - self.quad_corr_o_k[b]) * tanh_b_plus_minus
                                                   + (self.pair_corr_o_k[0, b] - self.pair_corr_o_k[1, b] + self.pair_corr_o_k[2, b]
                                                      - self.quad_corr_o_k[b]) * tanh_b_minus_plus
                                                   + (self.pair_corr_o_k[0, b] - self.pair_corr_o_k[1, b] - self.pair_corr_o_k[2, b]
                                                      + self.quad_corr_o_k[b]) * tanh_b_minus_minus) / 4 +
                                                  (1 - self.se_per_contribution) * pe_contribution[b])

    def simulate_mf_from_context(self, max_steps):
        # In order for this method to work properly, a simulate_mf() method has had to be run previously at least for
        # self.context_size steps

        # Initialize attention to the last computed attention
        att = self.mf_statistics["att"][self.context_size - 1, :]

        # We initialize the model at the end of the previous
        ini_t = self.context_size
        for t in range(ini_t, max_steps):
            self.compute_mf_optimized(t, att)
            att = self.attention_mf_optimized(t)

    def simulate_mf(self, x0, max_steps):
        self.x_list[0, :] = x0

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(x0, t=0)
        att = self.attention_mf_optimized(t=0)

        for t in range(1, max_steps):
            self.compute_mf_optimized(t, att)
            att = self.attention_mf_optimized(t)
