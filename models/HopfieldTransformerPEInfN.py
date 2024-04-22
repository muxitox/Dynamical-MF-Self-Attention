import copy
import numpy as np
from scipy.special import softmax


class HopfieldTransformerInfN:

    def __init__(self, beta_o, beta_att, num_feat_patterns, positional_embedding_bitsize, vocab, context_size,
                 max_sim_steps=512, min_saved_step=0, normalize_weights_str="1", reorder_weights=False,
                 correlations_from_weights=True, num_segments_corrs=3, pe_mode=0, semantic_embedding_bitsize=0,
                 se_per_contribution=0.95, gaussian_scale_str=None, compute_inf_normalization=True, N_normalization=None):

        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = semantic_embedding_bitsize
        self.pe_bit_size = positional_embedding_bitsize
        self.se_per_contribution = se_per_contribution
        self.vocab = vocab

        self.embedding_size = semantic_embedding_bitsize + positional_embedding_bitsize

        self.context_size = context_size
        self.num_feat_patterns = num_feat_patterns
        self.max_sim_steps = max_sim_steps
        self.min_saved_step = min_saved_step
        self.num_saved_steps = max_sim_steps - min_saved_step

        self.N_normalization = N_normalization
        if N_normalization is None:
            self.N_normalization = semantic_embedding_bitsize

        self.N_normalization += positional_embedding_bitsize


        self.run_exact_inf = compute_inf_normalization
        N = self.N_normalization
        M = num_feat_patterns
        self.normalize_weights_str = normalize_weights_str
        self.inf_normalization = self.define_normalization_inf()

        self.se_bit_size = 99
        self.embedding_size = semantic_embedding_bitsize + positional_embedding_bitsize


        # Dynamically compute the normalize_weights_str string
        try:
            exec_str = f"self.normalizing_constant = {normalize_weights_str}"
            exec_str2 = f"self.gaussian_scale = {gaussian_scale_str}"
            exec(exec_str)
            exec(exec_str2)
        except:
            print("Either the exec_str for the normalizing_constant or for the gaussian_scale are not well defined")
            raise

        self.W = np.zeros((num_feat_patterns, self.embedding_size))
        self.W_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1
        self.W[:, :self.se_bit_size] = self.W_SE
        self.W[:, -self.pe_bit_size:] = self.W_SE[:, -self.pe_bit_size:]

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

            self.Wo = np.zeros((num_feat_patterns, self.embedding_size))
            self.Wv = np.zeros((num_feat_patterns, self.embedding_size))
            self.Wq = np.zeros((num_feat_patterns, self.embedding_size))
            self.Wk = np.zeros((num_feat_patterns, self.embedding_size))

            self.Wo_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wv_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wq_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wk_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1

            self.Wo[:, :self.se_bit_size] = self.Wo_SE
            self.Wv[:, :self.se_bit_size] = self.Wv_SE
            self.Wq[:, :self.se_bit_size] = self.Wq_SE
            self.Wk[:, :self.se_bit_size] = self.Wk_SE

            if pe_mode == 1 or (pe_mode == 0 and correlations_from_weights == 3):
                # If pe_mode==0 and correlations_from_weighst=3. We use this (its like setting them random below but found seeds are more interesting)
                self.Wo[:, -self.pe_bit_size:] = self.Wo_SE[:, -self.pe_bit_size:]
                self.Wv[:, -self.pe_bit_size:] = self.Wv_SE[:, -self.pe_bit_size:]
                self.Wq[:, -self.pe_bit_size:] = self.Wq_SE[:, -self.pe_bit_size:]
                self.Wk[:, -self.pe_bit_size:] = self.Wk_SE[:, -self.pe_bit_size:]
            elif pe_mode == 0:
                self.Wo[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wv[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wq[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wk[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1

            self.W = self.Wo

            matrix_list = [self.Wo, self.Wv, self.Wq, self.Wk]

        if correlations_from_weights == 3:  #  create uniform +1 -1 segments and combine them

            segment_size = semantic_embedding_bitsize / num_segments_corrs

            pe_num_segments = int(positional_embedding_bitsize / segment_size) + 1
            segments_diff = num_segments_corrs - pe_num_segments

            for curr_W in matrix_list:
                for i in range(0, num_feat_patterns):
                    for segment_id in range(0, num_segments_corrs):
                        plus_minus_one = np.random.randint(2, size=1) * 2 - 1

                        segment_begin = int(segment_id * segment_size)
                        segment_end = int(segment_begin + segment_size)
                        curr_W[i, segment_begin:segment_end] = plus_minus_one  # Initialize that segment randomly to +-1

                    if pe_mode == 1:
                        # We want the positional encoding to be equal right to left to the segments
                        for pe_segment_id in range(0, pe_num_segments):
                            segment_end_pe = int(self.embedding_size - pe_segment_id * segment_size + 1)
                            segment_begin_pe = max(semantic_embedding_bitsize, int(positional_embedding_bitsize -
                                                                                   (pe_segment_id + 1) * segment_size))

                            segment_begin = int((pe_segment_id + segments_diff) * segment_size)

                            curr_W[i, segment_begin_pe:segment_end_pe] = curr_W[
                                i, segment_begin]  # Initialize PE to its corresponding segment

        if correlations_from_weights == 1 or correlations_from_weights == 3:  # Create matrices and compute correlations from them

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
            self.pair_corr_o_k /= semantic_embedding_bitsize
            self.pair_corr_o_q /= semantic_embedding_bitsize

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
                self.quad_corr_o_k /= semantic_embedding_bitsize

        elif correlations_from_weights == 0:  #  Normal weights and 4 correlations come from individual corrs
            sc = self.gaussian_scale
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
                # self.quad_corr_o_o = np.random.normal(0, sc, num_feat_patterns)
                # self.quad_corr_o_v = np.random.normal(0, sc, num_feat_patterns)
                # self.quad_corr_o_k = np.random.normal(0, sc, num_feat_patterns)
                # self.quad_corr_o_q = np.random.normal(0, sc, num_feat_patterns)
                #
                # self.quad_corr_o_o = np.clip(self.three_corr_o_o, -1, 1)
                # self.quad_corr_o_v = np.clip(self.three_corr_o_v, -1, 1)
                # self.quad_corr_o_k = np.clip(self.three_corr_o_k, -1, 1)
                # self.quad_corr_o_q = np.clip(self.three_corr_o_q, -1, 1)

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
                    Wo_corr_mo = self.corr_mo[(b + 1) % num_feat_patterns] * self.corr_mo[(b + 2) % num_feat_patterns]
                    self.quad_corr_o_o[b] += Wo_corr_mo
                    self.quad_corr_o_v[b] += Wo_corr * self.corr_mv[b]
                    self.quad_corr_o_k[b] += Wo_corr * self.corr_mk[b]
                    self.quad_corr_o_q[b] += Wo_corr * self.corr_mq[b]

        self.even_corr_o_o = self.pair_corr_o_o
        self.even_corr_o_v = self.pair_corr_o_v
        self.even_corr_o_k = self.pair_corr_o_k
        self.even_corr_o_q = self.pair_corr_o_q

        if num_feat_patterns == 1:
            self.sign_matrix = np.ones((1, 1))  # Create empty dimension in order to allow matrix indexing later
        elif num_feat_patterns == 2:
            self.sign_matrix = np.array([[1, 1], [1, -1]])
        elif num_feat_patterns == 3:
            self.sign_matrix = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])

            self.even_corr_o_o = np.vstack((self.pair_corr_o_o, self.quad_corr_o_o))
            self.even_corr_o_v = np.vstack((self.pair_corr_o_v, self.quad_corr_o_v))
            self.even_corr_o_k = np.vstack((self.pair_corr_o_k, self.quad_corr_o_k))
            self.even_corr_o_q = np.vstack((self.pair_corr_o_q, self.quad_corr_o_q))

        # Pre-compute the signs of the correlations
        self.corr_signed_o = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_o)
        self.corr_signed_v = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_v)
        self.corr_signed_q = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_q)
        self.corr_signed_k = np.einsum("ja,ab->jb", self.sign_matrix, self.even_corr_o_k)


        self.mv_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.mq_window = np.zeros(self.num_feat_patterns)
        self.mk_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.att_window = np.zeros(self.num_feat_patterns)

        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        # Create variables for saving the statistics of the mean-field model
        self.mf_statistics = {}
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, num_feat_patterns))

    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def reset_data(self):

        self.mv_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.mq_window = np.zeros(self.num_feat_patterns)
        self.mk_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.att_window = np.zeros(self.num_feat_patterns)

        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, self.num_feat_patterns))

    def reset_data_keep_context(self):

        # Shift values so the first value erased in the context window is the oldest one
        shift_amount = self.context_size - self.context_index - 1
        self.shift_d_window(shift_amount)

        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, self.num_feat_patterns))


    def qk_f_mf(self, tau):

        mqk = self.mq_window @ self.mk_window["mk"][tau]
        return self.beta_att * mqk

    def softmax(self, key_prob_unnorm, effective_context_size):

        if self.run_exact_inf:
            # In infty the softmax saturates and evolves into argmax
            max_val = max(key_prob_unnorm)
            max_ids = np.argwhere(key_prob_unnorm == max_val)
            selected_mvs = self.mv_window[max_ids]

            # The array created has 1 more empty dimension than we need, so we index by 0
            self.att_window = np.mean(selected_mvs, axis=0)[0]
            # We'll deal with normalization in the mf_computation function

        else:
            key_prob = softmax(self.N_normalization**2 * key_prob_unnorm / self.normalizing_constant)
            self.att_window = self.N_normalization * self.mv_window[:effective_context_size].T @ key_prob

    def attention_mf_unoptimized(self, t):

        effective_context_size = min(self.context_size, t + 1)

        key_prob_unnorm = np.zeros(effective_context_size)
        for tau in range(0, effective_context_size):
            key_prob_unnorm[tau] = self.qk_f_mf(tau)

        self.softmax(key_prob_unnorm, effective_context_size)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        if t >= self.min_saved_step:
            self.mf_statistics["att"][t - self.min_saved_step] = copy.deepcopy(self.att_window)

    def attention_mf(self, t):

        effective_context_size = min(self.context_size, t + 1)

        mqk = np.einsum('b,tb -> t', self.mq_window, self.mk_window[:effective_context_size],
                        optimize=True)

        key_prob_unnorm = self.beta_att * mqk
        self.softmax(key_prob_unnorm, effective_context_size)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        if t >= self.min_saved_step:
            self.mf_statistics["att"][t - self.min_saved_step] = copy.deepcopy(self.att_window)

    # def attention_mf_optimized(self, t):
    #
    #     effective_context_size = min(self.context_size, t + 1)
    #
    #     mqk = np.einsum('b,tb -> t', self.mq_window, self.mk_window[:effective_context_size],
    #                     optimize=True)
    #
    #     key_prob_unnorm = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk
    #     key_prob = softmax(key_prob_unnorm)
    #
    #     self.att_window = self.embedding_size * (self.mv_window[:effective_context_size].T @ key_prob)


    def save_stats(self, t, mo, mo_se, mv, mq, mk):
        index_t = t - self.min_saved_step
        self.mf_statistics["mo"][index_t] = copy.deepcopy(mo)
        self.mf_statistics["mo_se"][index_t] = copy.deepcopy(mo_se)
        self.mf_statistics["mv"][index_t] = copy.deepcopy(mv)
        self.mf_statistics["mq"][index_t] = copy.deepcopy(mq)
        self.mf_statistics["mk"][index_t] = copy.deepcopy(mk)

    def compute_means_from_data(self, x0, t):

        self.context_index = t % self.context_size

        self.mv_window[self.context_index, :] = x0 @ self.Wv.T / self.embedding_size
        self.mq_window = x0 @ self.Wq.T / self.embedding_size
        self.mk_window[self.context_index, :] = x0 @ self.Wk.T / self.embedding_size

        if t >= self.min_saved_step:
            mo = x0 @ self.Wo.T / self.embedding_size
            mo_se = x0[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.se_bit_size

            self.save_stats(t, mo, mo_se, self.mv_window[self.context_index, :], self.mq_window,
                            self.mk_window[self.context_index, :])

    def define_normalization_inf(self):
        if self.normalize_weights_str == "N":
            total_normalization = 1
        elif self.normalize_weights_str == "N*M" or self.normalize_weights_str == "M*N":
            total_normalization = 1 / self.num_feat_patterns
        elif self.normalize_weights_str == "N*np.sqrt(M)" or self.normalize_weights_str == "np.sqrt(M)*N":
            total_normalization = 1 / np.sqrt(self.num_feat_patterns)
        else: # We are asuming normalization constant U < N in this case
            total_normalization = np.inf

        return total_normalization

    def compute_mf(self, t):

        att = self.att_window

        pos_vec = self.vocab.encode_pos(t % self.context_size)
        pe_contribution_o = np.einsum('bi,i ->b', self.Wo[:, self.se_bit_size:],
                                      pos_vec, optimize=True) / self.pe_bit_size

        pe_contribution_v = np.einsum('bi,i ->b', self.Wv[:, self.se_bit_size:],
                                      pos_vec, optimize=True) / self.pe_bit_size

        pe_contribution_q = np.einsum('bi,i ->b', self.Wq[:, self.se_bit_size:],
                                      pos_vec, optimize=True) / self.pe_bit_size

        pe_contribution_k = np.einsum('bi,i ->b', self.Wk[:, self.se_bit_size:],
                                      pos_vec, optimize=True) / self.pe_bit_size


        if self.run_exact_inf:
            # In infty, we are going to deal with the order of N in the attention divided by U


            sign_att_patterns = self.beta_o *  np.einsum("jb,b->j",
                                                                 self.sign_matrix[:, :self.num_feat_patterns],
                                                                 att)

            if not np.allclose(sign_att_patterns, 0):
                # If result is not 0, normalize by inf
                sign_att_patterns *= self.inf_normalization
            tanh_j_signs = np.tanh(sign_att_patterns)

        else:
            tanh_j_signs = np.tanh(
                self.beta_o * (1 / self.normalizing_constant) * np.einsum("jb,b->j",
                                                                          self.sign_matrix[:, :self.num_feat_patterns],
                                                                          att))

        self.context_index = t % self.context_size

        mv_se = (self.se_per_contribution * np.einsum("jb,j->b", self.corr_signed_v, tanh_j_signs)
                 / 2 ** (self.num_feat_patterns - 1))
        self.mv_window[self.context_index] = mv_se + (1 - self.se_per_contribution) * pe_contribution_v

        mq_se = (self.se_per_contribution * np.einsum("jb,j->b", self.corr_signed_q, tanh_j_signs)
                 / 2 ** (self.num_feat_patterns - 1))
        self.mq_window = mq_se + (1 - self.se_per_contribution) * pe_contribution_q

        mk_se = (self.se_per_contribution * np.einsum("jb,j->b", self.corr_signed_k, tanh_j_signs)
                 / 2 ** (self.num_feat_patterns - 1))
        self.mk_window[self.context_index] = mk_se + (1 - self.se_per_contribution) * pe_contribution_k

        if t >= self.min_saved_step:
            mo_se = (np.einsum("jb,j->b", self.corr_signed_o, tanh_j_signs)
                     / 2 ** (self.num_feat_patterns - 1))
            mo = (self.se_per_contribution * mo_se + (1 - self.se_per_contribution) * pe_contribution_o)

            self.save_stats(t, mo, mo_se, self.mv_window[self.context_index, :], self.mq_window,
                            self.mk_window[self.context_index, :])


    def shift_d_window(self, shift):
        # Roll the window of d copies by "shit" positions
        self.mv_window = np.roll(self.mv_window, shift, axis=0)
        self.mk_window = np.roll(self.mk_window, shift, axis=0)

    def simulate_mf_from_context(self, max_steps):
        # In order for this method to work properly, a simulate_mf() method has had to be run previously at least for
        # self.context_size steps

        # We have in self.att_window the last attention value

        # We initialize the model at the end of the previous execution
        ini_t = self.context_size
        for t in range(ini_t, max_steps):
            self.compute_mf(t)
            self.attention_mf(t)

    def simulate_mf(self, x0, max_steps):

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(x0, t=0)
        self.attention_mf(t=0)

        for t in range(1, max_steps):
            self.compute_mf(t)
            self.attention_mf(t)

    # def compute_mf_unoptimized(self, t):
    #
    #     att = self.att_window
    #     pos_vec = self.vocab.encode_pos(t % self.context_size)
    #     pe_contribution_o = np.einsum('bi,i ->b', self.Wo[:, self.se_bit_size:],
    #                                   pos_vec, optimize=True) / self.pe_bit_size
    #
    #     pe_contribution_v = np.einsum('bi,i ->b', self.Wv[:, self.se_bit_size:],
    #                                   pos_vec, optimize=True) / self.pe_bit_size
    #
    #     pe_contribution_q = np.einsum('bi,i ->b', self.Wq[:, self.se_bit_size:],
    #                                   pos_vec, optimize=True) / self.pe_bit_size
    #
    #     pe_contribution_k = np.einsum('bi,i ->b', self.Wk[:, self.se_bit_size:],
    #                                   pos_vec, optimize=True) / self.pe_bit_size
    #
    #     if self.num_feat_patterns == 1:
    #         tanh_b = np.tanh(self.beta_o * (1 / self.normalizing_constant) * att[0])
    #
    #         self.mf_statistics["mo_se"][t] = self.se_per_contribution * self.pair_corr_o_o * tanh_b
    #         self.mf_statistics["mo"][t] = self.mf_statistics["mo_se"][t] + (1 - self.se_per_contribution) * pe_contribution_o
    #         self.mf_statistics["mv"][t] = (self.se_per_contribution * self.pair_corr_o_v * tanh_b +
    #                                        (1 - self.se_per_contribution) * pe_contribution_v)
    #         self.mf_statistics["mq"][t] = (self.se_per_contribution * self.pair_corr_o_q * tanh_b +
    #                                        (1 - self.se_per_contribution) * pe_contribution_q)
    #         self.mf_statistics["mk"][t] = (self.se_per_contribution * self.pair_corr_o_k * tanh_b +
    #                                        (1 - self.se_per_contribution) * pe_contribution_k)
    #
    #     elif self.num_feat_patterns == 2:
    #
    #         tanh_b_plus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] + att[1]))
    #         tanh_b_minus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] - att[1]))
    #
    #         for b in range(0, self.num_feat_patterns):
    #
    #             self.mf_statistics["mo_se"][t, b] = (self.se_per_contribution *
    #                         ((self.pair_corr_o_o[0,b] + self.pair_corr_o_o[1,b]) * tanh_b_plus
    #                          + (self.pair_corr_o_o[0,b] - self.pair_corr_o_o[1,b]) * tanh_b_minus) / 2 )
    #
    #             self.mf_statistics["mo"][t, b] = (self.mf_statistics["mo_se"][t,b] +
    #                                               (1 - self.se_per_contribution) * pe_contribution_o[b])
    #
    #             self.mf_statistics["mv"][t, b] = (self.se_per_contribution *
    #                         ((self.pair_corr_o_v[0,b] + self.pair_corr_o_v[1,b]) * tanh_b_plus +
    #                          (self.pair_corr_o_v[0,b] - self.pair_corr_o_v[1,b]) * tanh_b_minus ) / 2
    #                             + (1 - self.se_per_contribution) * pe_contribution_v[b])
    #
    #             self.mf_statistics["mq"][t, b] = (self.se_per_contribution *
    #                         ((self.pair_corr_o_q[0, b] + self.pair_corr_o_q[1, b]) * tanh_b_plus +
    #                          (self.pair_corr_o_q[0, b] - self.pair_corr_o_q[1, b]) * tanh_b_minus ) / 2
    #                             + (1 - self.se_per_contribution) * pe_contribution_q[b])
    #
    #             self.mf_statistics["mk"][t, b] = (self.se_per_contribution *
    #                         ((self.pair_corr_o_k[0, b] + self.pair_corr_o_k[1, b]) * tanh_b_plus +
    #                          (self.pair_corr_o_k[0, b] - self.pair_corr_o_k[1, b]) * tanh_b_minus ) / 2
    #                           + (1 - self.se_per_contribution) * pe_contribution_k[b])
    #
    #     else:
    #         tanh_b_plus_plus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] + att[1] + att[2]))
    #         tanh_b_plus_minus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] + att[1] - att[2]))
    #         tanh_b_minus_plus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] - att[1] + att[2]))
    #         tanh_b_minus_minus = np.tanh(self.beta_o * (1 / self.normalizing_constant) * (att[0] - att[1] - att[2]))
    #
    #         for b in range(0, self.num_feat_patterns):
    #             self.mf_statistics["mo_se"][t, b] = (self.se_per_contribution *
    #                                                  ((self.pair_corr_o_o[0, b] + self.pair_corr_o_o[1, b] + self.pair_corr_o_o[2, b] + self.quad_corr_o_o[b]) * tanh_b_plus_plus
    #                                                   + (self.pair_corr_o_o[0, b] + self.pair_corr_o_o[1, b] - self.pair_corr_o_o[2, b] - self.quad_corr_o_o[b]) * tanh_b_plus_minus
    #                                                   + (self.pair_corr_o_o[0, b] - self.pair_corr_o_o[1, b] + self.pair_corr_o_o[2, b] - self.quad_corr_o_o[b]) * tanh_b_minus_plus
    #                                                   + (self.pair_corr_o_o[0, b] - self.pair_corr_o_o[1, b] - self.pair_corr_o_o[2, b] +
    #                                                      self.quad_corr_o_o[b]) * tanh_b_minus_minus) / 4)
    #
    #             self.mf_statistics["mo"][t, b] = (self.mf_statistics["mo_se"][t,b] +
    #                                               (1 - self.se_per_contribution) * pe_contribution_o[b])
    #
    #             self.mf_statistics["mv"][t, b] = (self.se_per_contribution *
    #                                               ((self.pair_corr_o_v[0, b] + self.pair_corr_o_v[1, b] + self.pair_corr_o_v[2, b]
    #                                                 + self.quad_corr_o_v[b]) * tanh_b_plus_plus
    #                                                + (self.pair_corr_o_v[0, b] + self.pair_corr_o_v[1, b] - self.pair_corr_o_v[2, b]
    #                                                   - self.quad_corr_o_v[b]) * tanh_b_plus_minus
    #                                                + (self.pair_corr_o_v[0, b] - self.pair_corr_o_v[1, b] + self.pair_corr_o_v[2, b]
    #                                                   - self.quad_corr_o_v[b]) * tanh_b_minus_plus
    #                                                + (self.pair_corr_o_v[0, b] - self.pair_corr_o_v[1, b] - self.pair_corr_o_v[2, b]
    #                                                   + self.quad_corr_o_v[b]) * tanh_b_minus_minus) / 4 +
    #                                               (1 - self.se_per_contribution) * pe_contribution_v[b])
    #
    #             self.mf_statistics["mq"][t, b] = (self.se_per_contribution *
    #                                               ((self.pair_corr_o_q[0, b] + self.pair_corr_o_q[1, b] + self.pair_corr_o_q[2, b]
    #                                                 + self.quad_corr_o_q[b]) * tanh_b_plus_plus
    #                                                + (self.pair_corr_o_q[0, b] + self.pair_corr_o_q[1, b] - self.pair_corr_o_q[2, b]
    #                                                   - self.quad_corr_o_q[b]) * tanh_b_plus_minus
    #                                                + (self.pair_corr_o_q[0, b] - self.pair_corr_o_q[1, b] + self.pair_corr_o_q[2, b]
    #                                                   - self.quad_corr_o_q[b]) * tanh_b_minus_plus
    #                                                + (self.pair_corr_o_q[0, b] - self.pair_corr_o_q[1, b] - self.pair_corr_o_q[2, b]
    #                                                   + self.quad_corr_o_q[b]) * tanh_b_minus_minus) / 4 +
    #                                               (1 - self.se_per_contribution) * pe_contribution_q[b])
    #
    #             self.mf_statistics["mk"][t, b] = (self.se_per_contribution *
    #                                               ((self.pair_corr_o_k[0, b] + self.pair_corr_o_k[1, b] + self.pair_corr_o_k[2, b]
    #                                                 + self.quad_corr_o_k[b]) * tanh_b_plus_plus
    #                                                + (self.pair_corr_o_k[0, b] + self.pair_corr_o_k[1, b] - self.pair_corr_o_k[2, b]
    #                                                   - self.quad_corr_o_k[b]) * tanh_b_plus_minus
    #                                                + (self.pair_corr_o_k[0, b] - self.pair_corr_o_k[1, b] + self.pair_corr_o_k[2, b]
    #                                                   - self.quad_corr_o_k[b]) * tanh_b_minus_plus
    #                                                + (self.pair_corr_o_k[0, b] - self.pair_corr_o_k[1, b] - self.pair_corr_o_k[2, b]
    #                                                   + self.quad_corr_o_k[b]) * tanh_b_minus_minus) / 4 +
    #                                               (1 - self.se_per_contribution) * pe_contribution_k[b])