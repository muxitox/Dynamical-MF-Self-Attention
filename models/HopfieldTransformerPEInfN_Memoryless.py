import copy
import numpy as np
from scipy.special import softmax

class HopfieldTransformerInfNML:

    def __init__(self, beta_o, beta_att, num_feat_patterns, positional_embedding_bitsize, vocab, context_size, max_sim_steps=512,
                 normalize_weights_str="1", reorder_weights=False, correlations_from_weights=True, num_segments_corrs=3,
                 pe_mode=0, semantic_embedding_bitsize=0, se_per_contribution=0.95, gaussian_scale_str=None):

        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = semantic_embedding_bitsize
        self.pe_bit_size = positional_embedding_bitsize
        self.se_per_contribution = se_per_contribution
        self.vocab = vocab

        self.embedding_size = semantic_embedding_bitsize + positional_embedding_bitsize

        self.context_size = context_size

        N = self.embedding_size
        M = num_feat_patterns

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

            if pe_mode == 1 or (pe_mode==0 and correlations_from_weights == 3):
                # If pe_mode==0 and correlations_from_weighst=3. We use this (its like setting them random below but found seeds are more interesting)
                self.Wo[:, -self.pe_bit_size:] = self.Wo_SE[:, -self.pe_bit_size:]
                self.Wv[:, -self.pe_bit_size:] = self.Wv_SE[:, -self.pe_bit_size:]
                self.Wq[:, -self.pe_bit_size:] = self.Wq_SE[:, -self.pe_bit_size:]
                self.Wk[:, -self.pe_bit_size:] = self.Wk_SE[:, -self.pe_bit_size:]
            elif pe_mode == 0:
                self.Wo[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wv[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wq[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wk[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1

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

                            segment_end_pe = int(self.embedding_size - pe_segment_id*segment_size + 1)
                            segment_begin_pe = max(semantic_embedding_bitsize, int(positional_embedding_bitsize -
                                                                                   (pe_segment_id+1)*segment_size))

                            segment_begin = int((pe_segment_id + segments_diff) * segment_size)

                            curr_W[i, segment_begin_pe:segment_end_pe] = curr_W[i, segment_begin]  # Initialize PE to its corresponding segment

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

        # Create variables for the memory-less version computations for the mean-fields and positional embeddings

        self.mf_statistics = {}
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]

        self.mf_statistics["mo"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mf_statistics["mo_se"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mf_statistics["mv"] = np.zeros((self.max_sim_steps, self.context_size, self.num_feat_patterns))
        self.mf_statistics["mq"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mf_statistics["mk"] = np.zeros((self.max_sim_steps, self.context_size, self.num_feat_patterns))
        self.mf_statistics["pe"] = np.zeros((self.max_sim_steps, self.context_size, self.pe_bit_size))
        self.mf_statistics["att"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))

        self.create_der_matrix()

    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def create_der_matrix(self):
        self.datt_dmv = np.zeros((self.max_sim_steps, self.context_size))
        self.datt_dmq = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns))
        self.datt_dmk = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns * self.context_size))

        self.dmo_dmo = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns))
        self.dmo_dmv = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns * self.context_size))
        self.dmo_dmq = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns))
        self.dmo_dmk = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns * self.context_size))
        self.dmo_dpe = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.pe_bit_size * self.context_size))

        self.dmv_dmo = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size, self.num_feat_patterns))
        self.dmv_dmv = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size,
                            self.num_feat_patterns * self.context_size))
        self.dmv_dmq = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size, self.num_feat_patterns))
        self.dmv_dmk = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size,
                            self.num_feat_patterns * self.context_size))
        self.dmv_dpe = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size,
                            self.pe_bit_size * self.context_size))

        self.dmq_dmo = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns))
        self.dmq_dmv = np.zeros((self.max_sim_steps, self.num_feat_patterns,
                            self.num_feat_patterns * self.context_size))
        self.dmq_dmq = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.num_feat_patterns))
        self.dmq_dmk = np.zeros((self.max_sim_steps, self.num_feat_patterns,
                            self.num_feat_patterns * self.context_size))
        self.dmq_dpe = np.zeros((self.max_sim_steps, self.num_feat_patterns, self.pe_bit_size * self.context_size))

        self.dmk_dmo = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size, self.num_feat_patterns))
        self.dmk_dmv = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size,
                            self.num_feat_patterns * self.context_size))
        self.dmk_dmq = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size, self.num_feat_patterns))
        self.dmk_dmk = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size,
                            self.num_feat_patterns * self.context_size))
        self.dmk_dpe = np.zeros((self.max_sim_steps, self.num_feat_patterns * self.context_size,
                            self.pe_bit_size * self.context_size))

        self.dpe_dmo = np.zeros((self.max_sim_steps, self.pe_bit_size * self.context_size, self.num_feat_patterns))  # This is zero
        self.dpe_dmv = np.zeros((self.max_sim_steps, self.pe_bit_size * self.context_size,
                            self.num_feat_patterns * self.context_size))                # This is zero
        self.dpe_dmq = np.zeros((self.max_sim_steps, self.pe_bit_size * self.context_size, self.num_feat_patterns))  # This is zero
        self.dpe_dmk = np.zeros((self.max_sim_steps, self.pe_bit_size * self.context_size,   # This is zero
                            self.num_feat_patterns * self.context_size))                # This is zero
        self.dpe_dpe = np.zeros((self.max_sim_steps, self.pe_bit_size * self.context_size,
                            self.pe_bit_size * self.context_size))

    def reset_data(self):

        self.mf_statistics["mo"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mf_statistics["mo_se"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mf_statistics["mv"] = np.zeros((self.max_sim_steps, self.context_size, self.num_feat_patterns))
        self.mf_statistics["mq"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mf_statistics["mk"] = np.zeros((self.max_sim_steps, self.context_size, self.num_feat_patterns))
        self.mf_statistics["pe"] = np.zeros((self.max_sim_steps, self.context_size, self.pe_bit_size))
        self.mf_statistics["att"] = np.zeros((self.max_sim_steps, self.num_feat_patterns))

        time_indices = list(range(0, self.context_size))
        time_indices.reverse()
        time_indices = np.roll(time_indices, 1)
        for d in range(0, self.context_size):
            # At d=0 we want position 1, not 0. Position 0 is already encoded
            self.mf_statistics["pe"][0, d, :] = self.vocab.encode_pos(time_indices[d] % self.context_size)

    def reset_data_keep_context(self):

        mf_statistics_copy = {}
        for name_i in self.statistics_names:
            mf_statistics_copy[name_i] = copy.deepcopy(self.mf_statistics[name_i])

        self.reset_data()

        for name_i in self.statistics_names:
            self.mf_statistics[name_i][self.context_size] = mf_statistics_copy[name_i][-1]

    def qk_f_mf(self, t, d):
        mqk = self.mf_statistics["mq"][t] @ self.mf_statistics["mk"][t, d, :]
        return self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk

    def attention_mf(self, t):

        effective_context_size = min(self.context_size, t + 1)

        key_prob = np.zeros(effective_context_size)
        for d in range(0, effective_context_size):
            key_prob[d] = self.qk_f_mf(t, d)
        key_prob = softmax(key_prob)

        att_t = self.embedding_size * (self.mf_statistics["mv"][t, :effective_context_size, :].T @ key_prob)

        self.mf_statistics["att"][t] = att_t

        return att_t

    def attention_mf_optimized(self, t):

        effective_context_size = min(self.context_size, t + 1)

        mqk = np.einsum('b,tb -> t', self.mf_statistics["mq"][t],
                        self.mf_statistics["mk"][t, :effective_context_size, :], optimize=True)

        key_prob = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk
        key_prob = softmax(key_prob)

        att_t = self.embedding_size * (self.mf_statistics["mv"][t,:effective_context_size].T @ key_prob)

        self.mf_statistics["att"][t] = att_t

        return att_t

    def compute_means_from_data(self, x, t):
        self.mf_statistics["mo"][t] = x @ self.Wo.T / self.embedding_size
        self.mf_statistics["mo_se"][t] = x[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.se_bit_size
        self.mf_statistics["mv"][t, 0, :] = x @ self.Wv.T / self.embedding_size
        self.mf_statistics["mq"][t] = x @ self.Wq.T / self.embedding_size
        self.mf_statistics["mk"][t, 0, :] = x @ self.Wk.T / self.embedding_size


    def shift_d_window(self, t):
        # Roll the window of d copies by 1 position
        self.mf_statistics["mv"][t] = np.roll(self.mf_statistics["mv"][t-1], 1, axis=0)
        self.mf_statistics["mk"][t] = np.roll(self.mf_statistics["mk"][t-1], 1, axis=0)
        self.mf_statistics["pe"][t] = np.roll(self.mf_statistics["pe"][t-1], 1, axis=0)

    def compute_mf_optimized(self, t, att):

        pos_vec = self.vocab.encode_pos(t % self.context_size)
        pe_contribution_o = np.einsum('bi,i ->b', self.Wo[:, self.se_bit_size:],
                                    pos_vec,
                                    optimize=True) / self.pe_bit_size

        pe_contribution_v = np.einsum('bi,i ->b', self.Wv[:, self.se_bit_size:],
                                    pos_vec,
                                    optimize=True) / self.pe_bit_size

        pe_contribution_q = np.einsum('bi,i ->b', self.Wq[:, self.se_bit_size:],
                                    pos_vec,
                                    optimize=True) / self.pe_bit_size

        pe_contribution_k = np.einsum('bi,i ->b', self.Wk[:, self.se_bit_size:],
                                    pos_vec,
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
        self.mf_statistics["mv"][t, 0, :] = mv_se + (1 - self.se_per_contribution) * pe_contribution_v

        mq_se = self.se_per_contribution * np.einsum("jb,j->b", corr_signed_q, tanh_j_sings) / 2 ** (
                    self.num_feat_patterns - 1)
        self.mf_statistics["mq"][t] = mq_se + (1 - self.se_per_contribution) * pe_contribution_q

        mk_se = self.se_per_contribution * np.einsum("jb,j->b", corr_signed_k, tanh_j_sings) / 2 ** (
                    self.num_feat_patterns - 1)
        self.mf_statistics["mk"][t, 0, :] = mk_se + (1 - self.se_per_contribution) * pe_contribution_k


    def simulate_mf_from_context(self, max_steps):
        # In order for this method to work properly, a simulate_mf() method has had to be run previously at least for
        # self.context_size steps

        # Initialize attention to the last computed attention
        att = self.mf_statistics["att"][self.context_size - 1, :]

        # We initialize the model at the end of the previous
        ini_t = self.context_size
        for t in range(ini_t, max_steps):
            self.shift_d_window(t)

            self.compute_mf_optimized(t, att)
            att = self.attention_mf_optimized(t)

    def der_att_dmv(self, t):

        effective_context_size = min(self.context_size, t + 1)

        mqk = np.einsum('b,tb -> t', self.mf_statistics["mq"][t],
                        self.mf_statistics["mk"][t, :effective_context_size, :], optimize=True)

        key_prob = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk
        key_prob = softmax(key_prob)

        datt_dmv = self.embedding_size * key_prob

        # The derivative of the attention wrt mv has the form (1,context_size) since the derivative is the same for
        # different b indices

        return datt_dmv


    def der_att_dmq(self, t):

        effective_context_size = min(self.context_size, t + 1)

        mqk = np.einsum('b,tb -> t', self.mf_statistics["mq"][t],
                        self.mf_statistics["mk"][t, :effective_context_size, :], optimize=True)

        key_prob = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk
        key_prob = softmax(key_prob)

        derv_part_12 = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
        for b in range(self.num_feat_patterns):
            for c in range(self.num_feat_patterns):
                for d in range(effective_context_size):
                    derv_part_12[b, c] += (self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 3 *
                                           (self.mf_statistics["mv"][t, d, b] * self.mf_statistics["mk"][t, d, c]
                                            * key_prob[d]))

        derv_part_22 = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
        for b in range(self.num_feat_patterns):
            for c in range(self.num_feat_patterns):
                for d in range(effective_context_size):
                    prod1 = self.embedding_size * (self.mf_statistics["mv"][t, d, b] * key_prob[d])
                    prod2 = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * (self.mf_statistics["mk"][t, d, c] * key_prob[d])
                    derv_part_22[b, c] += prod1 * prod2

        mvd_mkd_1 = np.einsum("di,dj->dij", self.mf_statistics["mk"][t, :effective_context_size],
                           self.mf_statistics["mv"][t, :effective_context_size])
        derv_part_1 = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 3 * mvd_mkd_1.T @ key_prob


        att_mk_t = (self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 *
                    (self.mf_statistics["mk"][t, :effective_context_size].T @ key_prob))

        att_mv_t = self.embedding_size * (self.mf_statistics["mv"][t, :effective_context_size].T @ key_prob)
        derv_part_2 = np.einsum("i,j->ij", att_mk_t, att_mv_t).T


        print("Test1", derv_part_1 - derv_part_12)
        print("Test2", derv_part_2 - derv_part_22)


        derv = derv_part_1 - derv_part_2
        print(derv)
        # The derivative of the attention wrt mv has the form (1,context_size) since the derivative is the same for
        # different b indices

        return derv  # Dimensions are (att_b, mq_c)

    def der_mf_d0(self, t):
        datt_dmv = self.der_att_dmv(t)
        datt_dmq = self.der_att_dmq(t)

    def compute_jacobians(self, t):
        self.der_mf_d0(t)

    def simulate_mf(self, x0, max_steps):

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(x0, t=0)
        att = self.attention_mf_optimized(t=0)
        self.compute_jacobians(t=0)

        for t in range(1, max_steps):
            self.shift_d_window(t)

            self.compute_mf_optimized(t, att)
            self.compute_jacobians(t)   # Compute jacobians of mf and PE at time t, wrt vars at t-1

            att = self.attention_mf_optimized(t)
