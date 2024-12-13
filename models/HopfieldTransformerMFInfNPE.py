import copy
import numpy as np
from autograd import numpy as anp
from models.TransformerBase import TransformerBase
from autograd import jacobian


class HopfieldTransformerMFInfNPE(TransformerBase):

    def __init__(self, beta_o, beta_att, num_feat_patterns, positional_embedding_bitsize, vocab, context_size,
                 max_sim_steps=512, min_saved_step=0, normalize_weights_str_att="N**2*np.sqrt(M)", normalize_weights_str_o="N",
                 reorder_weights=False, correlations_from_weights=True, num_segments_corrs=3, pe_mode=0,
                 semantic_embedding_bitsize=0, epsilon_pe=0.95, gaussian_scale_str=None,
                 compute_inf_normalization=True, N_normalization=None, scaling_o=1, scaling_att=1, jacobian=True):

        if num_feat_patterns < 1 or num_feat_patterns > 3:
            raise Exception("The number of patterns is neither 1, 2 or 3")

        if normalize_weights_str_att != "N**2*np.sqrt(M)" or normalize_weights_str_o != "N":
            raise Exception("Only N**2*np.sqrt(M) and N normalizaitons are implemented for the attention and output layers.")

        self.N_normalization = N_normalization
        if N_normalization is None:
            self.N_normalization = semantic_embedding_bitsize

        self.N_normalization += positional_embedding_bitsize

        TransformerBase.__init__(self, beta_o, beta_att, num_feat_patterns, positional_embedding_bitsize, vocab,
                                 context_size, self.N_normalization,
                                 max_sim_steps=max_sim_steps, min_saved_step=min_saved_step,
                                 normalize_weights_str_att=normalize_weights_str_att,
                                 normalize_weights_str_o=normalize_weights_str_o,
                                 reorder_weights=reorder_weights, pe_mode=pe_mode,
                                 semantic_embedding_bitsize=semantic_embedding_bitsize,
                                 scaling_o=scaling_o, scaling_att=scaling_att)

        # `se_per_contribution` must be defined like this for reproducibility of the paper results
        # Otherwise results differ a little for changes in small decimals
        self.se_per_contribution = 1 - epsilon_pe
        self.run_exact_inf = compute_inf_normalization
        self.normalize_weights_str_att = normalize_weights_str_att
        self.normalize_weights_str_o = normalize_weights_str_o
        self.inf_normalization_o = self.define_normalization_inf_o()
        self.inf_normalization_att = self.define_normalization_inf_att()

        if self.run_exact_inf:
            self.gamma = self.inf_normalization_att * self.beta_att * self.scaling_att

        # Dynamically compute the normalize_weights_str string
        try:
            exec_str = f"self.gaussian_scale = {gaussian_scale_str}"
            exec(exec_str)
        except:
            raise Exception("The exec_str for the gaussian_scale is not well defined")

        self.create_W_matrices(correlations_from_weights, num_segments_corrs)
        self.define_correlations(correlations_from_weights)

        # Create variables to save results
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        # Create variables for saving the statistics of the mean-field model
        self.mf_statistics = {}
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, num_feat_patterns))

        self.compute_jacobian = jacobian

        # Variables for accumulating the Lyapunov exponents and debugging
        self.lyapunov_size = self.num_feat_patterns * self.context_size + self.pe_bit_size * self.context_size
        self.S = np.zeros(self.lyapunov_size)
        self.S_i = np.zeros((self.num_saved_steps, self.lyapunov_size))
        self.S_i_sum = np.zeros((self.num_saved_steps, self.lyapunov_size))
        self.S_inf_flag = np.zeros(self.lyapunov_size)


    def create_W_matrices(self, correlations_from_weights, num_segments_corrs):
        self.W = np.zeros((self.num_feat_patterns, self.embedding_size))
        self.W_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
        self.W[:, :self.se_bit_size] = self.W_SE
        self.W[:, -self.pe_bit_size:] = self.W_SE[:, -self.pe_bit_size:]

        # Following, we have the code for initializating the weight matrices from which we compute the correlations.

        if self.reorder_weights:
            self.Wo = np.copy(self.W)
            np.random.shuffle(self.Wo)
            self.Wv = np.copy(self.W)
            np.random.shuffle(self.Wv)
            # self.Wv = np.roll(self.Wo, 1, 1)
            self.Wq = np.copy(self.W)
            np.random.shuffle(self.Wq)
            self.Wk = np.copy(self.W)
            np.random.shuffle(self.Wk)

        else:

            self.Wo_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wv_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wq_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wk_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1

            self.Wo[:, :self.se_bit_size] = self.Wo_SE
            self.Wv[:, :self.se_bit_size] = self.Wv_SE
            self.Wq[:, :self.se_bit_size] = self.Wq_SE
            self.Wk[:, :self.se_bit_size] = self.Wk_SE

            if self.pe_mode == 1 or (self.pe_mode == 0 and correlations_from_weights == 3):
                # If pe_mode==0 and correlations_from_weighst=3. We use this (its like setting them random below but found seeds are more interesting)
                self.Wo[:, -self.pe_bit_size:] = self.Wo_SE[:, -self.pe_bit_size:]
                self.Wv[:, -self.pe_bit_size:] = self.Wv_SE[:, -self.pe_bit_size:]
                self.Wq[:, -self.pe_bit_size:] = self.Wq_SE[:, -self.pe_bit_size:]
                self.Wk[:, -self.pe_bit_size:] = self.Wk_SE[:, -self.pe_bit_size:]
            elif self.pe_mode == 0:
                self.Wo[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(
                                                                   self.num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wv[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(
                                                                   self.num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wq[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(
                                                                   self.num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wk[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(
                                                                   self.num_feat_patterns, self.pe_bit_size)) * 2 - 1

            self.W = self.Wo

        matrix_list = [self.Wo, self.Wv, self.Wq, self.Wk]

        self.W_dict = {}
        self.features_names = ["o", "v", "q", "k"]
        self.W_dict["o"] = self.Wo
        self.W_dict["v"] = self.Wv
        self.W_dict["q"] = self.Wq
        self.W_dict["k"] = self.Wk

        if correlations_from_weights == 3:  # create uniform +1 -1 segments and combine them

            segment_size = self.se_bit_size / num_segments_corrs

            pe_num_segments = int(self.pe_bit_size / segment_size) + 1
            segments_diff = num_segments_corrs - pe_num_segments

            for curr_W in matrix_list:
                for i in range(0, self.num_feat_patterns):
                    for segment_id in range(0, num_segments_corrs):
                        plus_minus_one = np.random.randint(2, size=1) * 2 - 1

                        segment_begin = int(segment_id * segment_size)
                        segment_end = int(segment_begin + segment_size)
                        curr_W[i, segment_begin:segment_end] = plus_minus_one  # Initialize that segment randomly to +-1

                    if self.pe_mode == 1:
                        # We want the positional encoding to be equal right to left to the segments
                        for pe_segment_id in range(0, pe_num_segments):
                            segment_end_pe = int(self.embedding_size - pe_segment_id * segment_size + 1)
                            segment_begin_pe = max(self.se_bit_size, int(self.pe_bit_size -
                                                                         (pe_segment_id + 1) * segment_size))

                            segment_begin = int((pe_segment_id + segments_diff) * segment_size)

                            # Initialize PE to its corresponding segment
                            curr_W[i, segment_begin_pe:segment_end_pe] = curr_W[i, segment_begin]


    def define_correlations(self, correlations_from_weights):

        if correlations_from_weights == 1 or correlations_from_weights == 3:  # Compute correlations from created matrices

            # Create correlations from matrices for comparison
            self.define_pair_correlations_from_weights()
            self.define_quad_correlations_from_weights()

        elif correlations_from_weights == 0:  # Normal weights and 4 correlations come from individual corrs
            sc = self.gaussian_scale
            # Create random correlations
            self.pair_corr_o_o = np.random.normal(0, sc, (self.num_feat_patterns, self.num_feat_patterns))
            self.pair_corr_o_v = np.random.normal(0, sc, (self.num_feat_patterns, self.num_feat_patterns))
            self.pair_corr_o_k = np.random.normal(0, sc, (self.num_feat_patterns, self.num_feat_patterns))
            self.pair_corr_o_q = np.random.normal(0, sc, (self.num_feat_patterns, self.num_feat_patterns))

            # Set autocorrelations to 0
            np.fill_diagonal(self.pair_corr_o_o, 1)

            self.pair_corr_o_o = np.clip(self.pair_corr_o_o, -1, 1)
            self.pair_corr_o_v = np.clip(self.pair_corr_o_v, -1, 1)
            self.pair_corr_o_k = np.clip(self.pair_corr_o_k, -1, 1)
            self.pair_corr_o_q = np.clip(self.pair_corr_o_q, -1, 1)

            if self.num_feat_patterns == 3:
                self.quad_corr_o_o = np.prod(self.pair_corr_o_o, axis=0)
                self.quad_corr_o_v = np.prod(self.pair_corr_o_v, axis=0)
                self.quad_corr_o_k = np.prod(self.pair_corr_o_k, axis=0)
                self.quad_corr_o_q = np.prod(self.pair_corr_o_q, axis=0)

        elif correlations_from_weights == 2:  # Correlations from uniform means

            # Create initial means that will later define the correlations
            self.corr_mo = np.random.rand(self.num_feat_patterns)
            self.corr_mv = np.random.rand(self.num_feat_patterns)
            self.corr_mq = np.random.rand(self.num_feat_patterns)
            self.corr_mk = np.random.rand(self.num_feat_patterns)

            self.pair_corr_o_o = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
            self.pair_corr_o_v = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
            self.pair_corr_o_k = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
            self.pair_corr_o_q = np.zeros((self.num_feat_patterns, self.num_feat_patterns))

            for b in range(0, self.num_feat_patterns):
                for a in range(0, self.num_feat_patterns):
                    self.pair_corr_o_o[a, b] += self.corr_mo[a] * self.corr_mo[b]
                    self.pair_corr_o_v[a, b] += self.corr_mo[a] * self.corr_mv[b]
                    self.pair_corr_o_q[a, b] += self.corr_mo[a] * self.corr_mq[b]
                    self.pair_corr_o_k[a, b] += self.corr_mo[a] * self.corr_mk[b]

            np.fill_diagonal(self.pair_corr_o_o, 1)

            if self.num_feat_patterns == 3:
                self.quad_corr_o_o = np.zeros(self.num_feat_patterns)
                self.quad_corr_o_v = np.zeros(self.num_feat_patterns)
                self.quad_corr_o_k = np.zeros(self.num_feat_patterns)
                self.quad_corr_o_q = np.zeros(self.num_feat_patterns)

                for b in range(0, self.num_feat_patterns):
                    Wo_corr = self.corr_mo[0] * self.corr_mo[1] * self.corr_mo[2]
                    Wo_corr_mo = self.corr_mo[(b + 1) % self.num_feat_patterns] * self.corr_mo[
                        (b + 2) % self.num_feat_patterns]
                    self.quad_corr_o_o[b] += Wo_corr_mo
                    self.quad_corr_o_v[b] += Wo_corr * self.corr_mv[b]
                    self.quad_corr_o_k[b] += Wo_corr * self.corr_mk[b]
                    self.quad_corr_o_q[b] += Wo_corr * self.corr_mq[b]

        self.even_corr_o_o = copy.deepcopy(self.pair_corr_o_o)
        self.even_corr_o_v = copy.deepcopy(self.pair_corr_o_v)
        self.even_corr_o_k = copy.deepcopy(self.pair_corr_o_k)
        self.even_corr_o_q = copy.deepcopy(self.pair_corr_o_q)

        # Create matrix of signs to compute the signs of the attention
        if self.num_feat_patterns == 1:
            self.sign_matrix = np.ones((1, 1))  # Create empty dimension in order to allow matrix indexing later
        elif self.num_feat_patterns == 2:
            self.sign_matrix = np.array([[1, 1], [1, -1]])
        elif self.num_feat_patterns == 3:
            self.sign_matrix = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])

            self.even_corr_o_o = np.vstack((self.pair_corr_o_o, self.quad_corr_o_o))
            self.even_corr_o_v = np.vstack((self.pair_corr_o_v, self.quad_corr_o_v))
            self.even_corr_o_k = np.vstack((self.pair_corr_o_k, self.quad_corr_o_k))
            self.even_corr_o_q = np.vstack((self.pair_corr_o_q, self.quad_corr_o_q))

        self.even_corr = {}
        self.even_corr["o"] = self.even_corr_o_o
        self.even_corr["v"] = self.even_corr_o_v
        self.even_corr["q"] = self.even_corr_o_q
        self.even_corr["k"] = self.even_corr_o_k


        # Pre-compute the signs of the correlations
        # Per every pattern a, we have j combinations of correlations with other patterns (a,)
        self.corr_signed = {}
        self.corr_signed["o"] = np.einsum("jb,ba->ja", self.sign_matrix, self.even_corr_o_o)
        self.corr_signed["v"] = np.einsum("jb,ba->ja", self.sign_matrix, self.even_corr_o_v)
        self.corr_signed["q"] = np.einsum("jb,ba->ja", self.sign_matrix, self.even_corr_o_q)
        self.corr_signed["k"] = np.einsum("jb,ba->ja", self.sign_matrix, self.even_corr_o_k)


    def set_beta_o(self, beta_o):
        self.beta_o = beta_o

    def set_beta_att(self, beta_att):
        self.beta_att = beta_att

    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def set_epsilon_pe(self, epsilon_pe):
        self.epsilon_pe = epsilon_pe

    def reset_data(self):
        self.S = np.zeros(self.lyapunov_size)
        self.S_i = np.zeros((self.num_saved_steps, self.lyapunov_size))
        self.S_i_sum = np.zeros((self.num_saved_steps, self.lyapunov_size))

        self.effective_context_size = 0
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, self.num_feat_patterns))

    def get_context_window(self):
        index_t = self.t - self.min_saved_step + 1

        att = copy.deepcopy(self.mf_statistics["att"][index_t-self.context_size:index_t][::-1])
        mo = copy.deepcopy(self.mf_statistics["mo"][index_t-self.context_size:index_t][::-1])
        mv = copy.deepcopy(self.mf_statistics["mv"][index_t-self.context_size:index_t][::-1])
        mq = copy.deepcopy(self.mf_statistics["mq"][index_t-self.context_size:index_t][::-1])
        mk = copy.deepcopy(self.mf_statistics["mk"][index_t-self.context_size:index_t][::-1])
        pe = copy.deepcopy(self.PE.p_t_d)

        return att, mo, mv, mq, mk, pe

    def save_att_stats(self, att):
        # Save stats in an array if the time threshold is surpassed
        if self.t >= self.min_saved_step:
            index_t = self.t - self.min_saved_step
            if isinstance(att, np.ndarray):
                self.mf_statistics["att"][index_t] = copy.deepcopy(att)
            else:
                self.mf_statistics["att"][index_t] = copy.deepcopy(att._value)


    def save_mf_stats(self, mo, mo_se, mv, mq, mk):
        # Save stats in an array if the time threshold is surpassed
        if self.t >= self.min_saved_step:

            index_t = self.t - self.min_saved_step
            if isinstance(mv, np.ndarray):
                self.mf_statistics["mo"][index_t] = copy.deepcopy(mo)
                self.mf_statistics["mo_se"][index_t] = copy.deepcopy(mo_se)
                self.mf_statistics["mv"][index_t] = copy.deepcopy(mv)
                self.mf_statistics["mq"][index_t] = copy.deepcopy(mq)
                self.mf_statistics["mk"][index_t] = copy.deepcopy(mk)
            else:
                self.mf_statistics["mo"][index_t] = copy.deepcopy(mo._value)
                self.mf_statistics["mo_se"][index_t] = copy.deepcopy(mo_se._value)
                self.mf_statistics["mv"][index_t] = copy.deepcopy(mv._value)
                self.mf_statistics["mq"][index_t] = copy.deepcopy(mq._value)
                self.mf_statistics["mk"][index_t] = copy.deepcopy(mk._value)

    def compute_means_from_data(self, x0, t, ponder_pe=True):
        # In the first version of the paper ponder_pe=False for the bif diagrams


        # Computes mean values from data. Basically used to compute m values from a x0 token.
        self.context_index = t % self.context_size

        if ponder_pe:

            mv = \
                ((self.se_per_contribution * x0[:self.se_bit_size] @ self.W_dict["v"][:, :self.se_bit_size].T / self.se_bit_size) +
                 ((1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["v"][:, -self.pe_bit_size:].T / self.pe_bit_size))

            mq = \
                ((self.se_per_contribution * x0[:self.se_bit_size] @ self.W_dict["q"][:, :self.se_bit_size].T / self.se_bit_size) +
                 ((1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["q"][:, -self.pe_bit_size:].T / self.pe_bit_size))

            mk = \
                ((self.se_per_contribution * x0[:self.se_bit_size] @ self.W_dict["k"][:, :self.se_bit_size].T / self.se_bit_size) +
                 ((1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["k"][:, -self.pe_bit_size:].T / self.pe_bit_size))

            if t >= self.min_saved_step:
                mo_se =  x0[:self.se_bit_size] @ self.W_dict["o"][:, :self.se_bit_size].T / self.se_bit_size

                mo = (self.se_per_contribution * mo_se +
                      (1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["o"][:, -self.pe_bit_size:].T / self.pe_bit_size)

                self.save_mf_stats(mo, mo_se, mv, mq, mk)

        else:
            mv = x0 @ self.W_dict["v"].T / self.embedding_size
            mq = x0 @ self.W_dict["q"].T / self.embedding_size
            mk = x0 @ self.W_dict["k"].T / self.embedding_size

            if t >= self.min_saved_step:
                mo = x0 @ self.W_dict["o"].T / self.embedding_size
                mo_se = x0[:self.se_bit_size] @ self.W_dict["o"][:, :self.se_bit_size].T / self.se_bit_size

                self.save_mf_stats(mo, mo_se, mv, mq, mk)

        return mv, mq, mk

    def create_mf_window_from_means(self, x0, ponder_pe=True):
        mv, mq, mk = self.compute_means_from_data(x0, t=0)

        mv = mv[np.newaxis, :]
        mq = mq[np.newaxis, :]
        mk = mk[np.newaxis, :]

        return mv, mq, mk

    def define_normalization_inf_o(self):
        # Define normalization values in infinity for the output
        if self.normalize_weights_str_o == "N":
            total_normalization = 1
        elif self.normalize_weights_str_o == "N*M" or self.normalize_weights_str_o == "M*N":
            total_normalization = 1 / self.num_feat_patterns
        elif self.normalize_weights_str_o == "N*np.sqrt(M)" or self.normalize_weights_str_o == "np.sqrt(M)*N":
            total_normalization = 1 / np.sqrt(self.num_feat_patterns)
        else:  # We are asuming normalization constant U < N in this case
            total_normalization = np.inf

        return total_normalization

    def define_normalization_inf_att(self):
        # Define normalization values in infinity for the self-attention
        if self.normalize_weights_str_att == "N**2":
            total_normalization = 1
        elif self.normalize_weights_str_att == "N**2*np.sqrt(M)":
            total_normalization = 1 / np.sqrt(self.num_feat_patterns)
        else:
            raise Exception("Define properly the attention normalization")
        return total_normalization

    def return_context_window(self):
        return self.att, self.mv_window, self.mq_window, self.mk_window

    @staticmethod
    def softmax(key_prob_unnorm):
        C = 10
        max_x = max(key_prob_unnorm)
        expp = anp.exp(key_prob_unnorm - max_x + C)
        sum_exp = anp.sum(expp)
        key_prob = expp / sum_exp

        return key_prob


    def attention(self, att_t_1_d, mv_window, mq, mk_window):

        # effective_context_size = min(self.context_size, self.t + 1)

        # Put in common queries and keys
        mqk = anp.einsum('b,tb -> t', mq[0], mk_window,
                         optimize=True)

        # For the scaling we assume that we are working with a perfectly inf system
        # Scale
        key_prob_unnorm = self.gamma * mqk

        # Compute softmax and average by mv
        # Assume perfectly inf system with self.normalize_weights_str_att == "N**2*np.sqrt(M)"
        key_prob = self.softmax(key_prob_unnorm)

        # We'll deal with normalization in the mf_computation function, but since we are returning
        # the average of mvs and not N*mv, we are already normalizing by a /N factor
        att_t_0 = anp.einsum("da,d->a", mv_window, key_prob)

        # Append new attention values to old ones
        att_t_d = anp.vstack((att_t_0, att_t_1_d[:self.context_size-1]))

        # Save att if required
        self.save_att_stats(att_t_0)

        return att_t_d

    def compute_mfs(self, att_t_d, p_t_d):

        # Positional Embedding contribution
        pe_contribution = {}
        for feat_name in self.features_names:

            effective_p = p_t_d[:self.effective_context_size]
            if feat_name == "q" or feat_name == "o":
                effective_p = p_t_d[np.newaxis, 0]  # Add new dimension to keep the einsum expresion simple

            pe_contribution[feat_name] = ((1 - self.se_per_contribution) *
                                          anp.einsum('bi,di ->db', self.W_dict[feat_name][:, self.se_bit_size:],
                                                    effective_p, optimize=True) / self.pe_bit_size)

        # We assume we are using the inf implementation
        # In infty, we are going to deal with the order of N in the output.

        sign_att_patterns = (self.beta_o * self.scaling_o *
                             anp.einsum("jb,db->dj", self.sign_matrix[:, :self.num_feat_patterns],
                                        att_t_d))

        # idx_not_zero = anp.where(sign_att_patterns != 0)
        # If result is not 0, normalize by inf (avoids NaNs)
        sign_att_patterns *= self.inf_normalization_o
        tanh_j_signs = anp.tanh(sign_att_patterns)

        # Compute the semantic part of every mean field needed for the attention
        m_alpha_se = {}
        for feat_name in self.features_names:
            # corr_signed has shape (num_combinations_signs, num_features_a).
            # For every feature a, you put together all the j combinations of signs

            tanh_j_signs_loop = tanh_j_signs
            if feat_name == "q" or feat_name == "o":
                # With "o" or "q" we just work with the current time-step.
                # "o" is not needed for computation, just to save stats to plot the trajectory.
                # Add a new empty dimension to keep the einsum expression simple
                tanh_j_signs_loop = tanh_j_signs[np.newaxis, 0]

            m_alpha_se[feat_name] = (self.se_per_contribution * anp.einsum("ja,dj->da",
                                                                              self.corr_signed[feat_name],
                                                                              tanh_j_signs_loop)
                                     / 2 ** (self.num_feat_patterns - 1))

        mo_se = m_alpha_se["o"]  # Feature not used in the Jacobian computation
        mo = mo_se + pe_contribution["o"]  # Feature not used in the Jacobian computation
        mv_window = m_alpha_se["v"] + pe_contribution["v"]
        mq = m_alpha_se["q"] + pe_contribution["q"]
        mk_window = m_alpha_se["k"] + pe_contribution["k"]

        self.save_mf_stats(mo, mo_se, mv_window[0], mq, mk_window[0])

        return mv_window, mq, mk_window

    def _step(self, input):

        # Reshape input shape into one more easily manageable for computing
        att_size = self.effective_context_size * self.num_feat_patterns
        att_t_1_d = input[:att_size]
        att_t_1_d = anp.reshape(att_t_1_d, (self.effective_context_size, self.num_feat_patterns))
        p_t_1_d = input[att_size:]
        p_t_1_d = anp.reshape(p_t_1_d, (self.context_size, self.pe_bit_size))

        # Compute mf values from previous attention values
        mv_window, mq, mk_window = self.compute_mfs(att_t_1_d, p_t_1_d)

        # Compute new attention values from previous attention values (shifted) and computed mean-fields
        att_t_d = self.attention(att_t_1_d, mv_window, mq, mk_window)

        # Compute Positional Encoding update
        p_t_d = self.PE.next_step_autograd(p_t_1_d)

        # Flatten back so autograd can compute the Jacobian
        att_size = att_t_d.shape[0] * self.num_feat_patterns
        att_t_d = anp.reshape(att_t_d, att_size)
        p_t_d = anp.reshape(p_t_d, self.context_size * self.pe_bit_size)

        output = anp.concatenate((att_t_d, p_t_d))

        # We return the output so autograd can compute the gradient, but to save computation time and avoid
        # repeating executions we also create this variable here.
        if isinstance(output, np.ndarray):
            self.next_input = copy.deepcopy(output)
        else:
            self.next_input = copy.deepcopy(output._value)

        return output


    def lyapunov_step(self, input, dx):

        # output = self._step(input)  # Output for testing
        J = self.Jacobian_Func(input)

        S_idx = self.t - self.min_saved_step

        # Compute perturbation
        dx = np.matmul(J, dx)
        # Decompose perturbation
        Q, R = np.linalg.qr(dx)
        d_exp = np.absolute(np.diag(R))
        dS = np.log(d_exp)
        # Q is orthogonal so we can use it for the next step
        dx = Q

        # Flag digits where log(0) has been computed
        flag_1 = np.copy(d_exp)
        flag_1[flag_1 > 0]  = 1
        self.S_inf_flag = np.logical_or(np.logical_not(flag_1), self.S_inf_flag)


        self.S += dS
        self.S_i[S_idx] = dS
        self.S_i_sum[S_idx] = copy.deepcopy(self.S)

        return dx

    def lyapunov_end(self):

        # Average by trajectory length
        self.S /= self.num_saved_steps

    def simulate(self, context_att, context_pe, max_steps, compute_lyapunov=True):
        # We simulate given a previously computed attention window and positional encoding

        ini_t = context_att.shape[0]
        self.t = ini_t

        # Flatten input for the jacobian
        att_t_1_d_flat = anp.reshape(context_att, context_att.shape[0] * self.num_feat_patterns)
        p_t_1_d_flat = anp.reshape(context_pe, self.context_size * self.pe_bit_size)
        self.next_input = np.concatenate((att_t_1_d_flat, p_t_1_d_flat))

        if compute_lyapunov:
            # Define here the Jacobian handler over the function _step()
            self.Jacobian_Func = jacobian(self._step)
            dx = np.eye(self.num_feat_patterns * self.context_size + self.pe_bit_size * self.context_size)

        for t in range(ini_t, max_steps):
            self.t = t
            self.effective_context_size = min(self.context_size, self.t)

            if not compute_lyapunov or (compute_lyapunov and (t < self.min_saved_step)):
                # If we don't want the gradients, just compute the output
                # _step() returns an output, but we also save it as self.next_input to save computation when
                # computing the gradients later
                self._step(self.next_input)

            if compute_lyapunov and (t >= self.min_saved_step):
                # Otherwise compute gradients and perturbations
                dx = self.lyapunov_step(self.next_input, dx)

        if compute_lyapunov:
            self.lyapunov_end()

    def simulate_from_token(self, x0, max_steps, compute_lyapunov=False):

        self.t = 0
        # Initialize attention with the info from the initial token
        mv, mq, mk = self.create_mf_window_from_means(x0)
        # Create empty array for appending attention values
        att_t_d = np.array([]).reshape(0, self.num_feat_patterns)
        # Create attention
        att_t_d = self.attention(att_t_d, mv, mq, mk)
        # Initialize rotating PE
        p_t_d = self.PE.initialize_rotating_pe()

        self.simulate(att_t_d, p_t_d, max_steps, compute_lyapunov)

