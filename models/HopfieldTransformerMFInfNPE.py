import copy
import numpy as np
from autograd import numpy as anp
from models.TransformerBase import TransformerBase
import scipy
from scipy.linalg import lu
from models.PositionalEncoding import PositionalEncoding
from autograd import grad, jacobian


class HopfieldTransformerMFInfNPE(TransformerBase):

    def __init__(self, beta_o, beta_att, num_feat_patterns, positional_embedding_bitsize, vocab, context_size,
                 max_sim_steps=512, min_saved_step=0, normalize_weights_str_att="N**2", normalize_weights_str_o="N",
                 reorder_weights=False, correlations_from_weights=True, num_segments_corrs=3, pe_mode=0,
                 semantic_embedding_bitsize=0, epsilon_pe=0.95, gaussian_scale_str=None,
                 compute_inf_normalization=True, N_normalization=None, scaling_o=1, scaling_att=1, jacobian=True):

        if num_feat_patterns < 1 or num_feat_patterns > 3:
            raise Exception("The number of patterns is neither 1, 2 or 3")

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

        # Create variable for the context window in the attention
        self.mv_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.mq_window = np.zeros(self.num_feat_patterns)
        self.mk_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.att = np.zeros(self.num_feat_patterns)
        # att values at previous step
        self.att_window = anp.zeros((self.context_size, self.num_feat_patterns))
        # Variable that puts together keys and queries and is already scaled with gamma
        self.key_prob_unnorm = np.zeros(self.context_size)


        # Create variables to save results
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        # Create variables for saving the statistics of the mean-field model
        self.mf_statistics = {}
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, num_feat_patterns))


        self.compute_jacobian = jacobian
        self.L = self.Lyapunov(self, positional_embedding_bitsize, vocab)


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


    class Lyapunov:
        def __init__(self, HT, positional_embedding_bitsize, vocab):

            self.HT = HT

            self.PE = PositionalEncoding(positional_embedding_bitsize, vocab, self.HT.context_size, K=10, type="tanh")

            # Variable for accumulating the Lyapunov exponents
            lyapunov_size = self.HT.num_feat_patterns * self.HT.context_size + self.HT.pe_bit_size * self.HT.context_size
            self.S = np.zeros(lyapunov_size)
            self.S_i = np.zeros((self.HT.num_saved_steps, lyapunov_size))
            self.S_i_sum = np.zeros((self.HT.num_saved_steps, lyapunov_size))

        @staticmethod
        def softmax(key_prob_unnorm):
            C = 10
            max_x = max(key_prob_unnorm)
            expp = anp.exp(key_prob_unnorm - max_x + C)
            sum_exp = anp.sum(expp)
            key_prob = expp / sum_exp

            return key_prob

        def key_averaging(self, key_prob_unnorm, mv_window):

            effective_context_size = min(self.HT.context_size, self.HT.t + 1)

            # Assume perfectly inf system with self.normalize_weights_str_att == "N**2*np.sqrt(M)"
            key_prob = self.softmax(key_prob_unnorm)

            # We'll deal with normalization in the mf_computation function, but since we are returning
            # the average of mvs and not N*mv, we are already normalizing by a /N factor
            att_t_0 = anp.einsum("da,d->a", mv_window[:effective_context_size], key_prob)

            return att_t_0


        def attention(self, att_t_1_d, mv_window, mq, mk_window):

            effective_context_size = min(self.HT.context_size, self.HT.t + 1)
            # Put in common queries and keys
            mqk = anp.einsum('b,tb -> t', mq, mk_window[:effective_context_size],
                            optimize=True)

            # For the scaling we assume that we are working with a perfectly inf system
            # Scale
            key_prob_unnorm = self.HT.gamma * mqk

            # Compute softmax and average by mv
            att_t_0 = self.key_averaging(key_prob_unnorm, mv_window)

            att_t_d = anp.vstack((att_t_0, att_t_1_d[:self.HT.context_size-1]))

            # att_t_d2 = anp.roll(att_t_1_d, 1, axis=0)
            # att_t_d2[0] = att_t_0

            return att_t_d

        def compute_mfs(self, att_t_d, p_t_d):

            effective_context_size = min(self.HT.context_size, self.HT.t + 1)

            # Positional Embedding contribution
            pe_contribution = {}
            for feat_name in self.HT.features_names:

                p_loop = p_t_d
                if feat_name == "q":
                    p_loop = p_t_d[np.newaxis, 0]  # Add new dimension to keep the einsum expresion simple

                pe_contribution[feat_name] = anp.einsum('bi,di ->db', self.HT.W_dict[feat_name][:, self.HT.se_bit_size:],
                                                       p_loop, optimize=True) / self.HT.pe_bit_size

            # We assume we are using the inf implementation
            # In infty, we are going to deal with the order of N in the output.

            sign_att_patterns = (self.HT.beta_o * self.HT.scaling_o *
                                 anp.einsum("jb,db->dj", self.HT.sign_matrix[:, :self.HT.num_feat_patterns],
                                           att_t_d[:effective_context_size]))

            # idx_not_zero = anp.where(sign_att_patterns != 0)
            # If result is not 0, normalize by inf (avoids NaNs)
            sign_att_patterns *= self.HT.inf_normalization_o
            tanh_j_signs = anp.tanh(sign_att_patterns)


            # Compute the semantic part of every mean field needed for the attention
            m_alpha_se = {}
            for feat_name in ["v", "q", "k"]:
                # corr_signed has shape (num_combinations_signs, num_features_a).
                # For every feature a, you put together all the j combinations of signs

                tanh_j_signs_loop = tanh_j_signs
                if feat_name == "q":
                    # With q we just work with the current time-step
                    tanh_j_signs_loop = tanh_j_signs[np.newaxis, 0] # Add new dimension to keep the einsum expresion simple


                m_alpha_se[feat_name] = (self.HT.se_per_contribution * anp.einsum("ja,dj->da",
                                                                              self.HT.corr_signed[feat_name],
                                                                              tanh_j_signs_loop)
                                         / 2 ** (self.HT.num_feat_patterns - 1))

            mv_window = m_alpha_se["v"] + (1 - self.HT.se_per_contribution) * pe_contribution["v"]
            mq = m_alpha_se["q"][0] + (1 - self.HT.se_per_contribution) * pe_contribution["q"][0] # Remove new axis
            mk_window = m_alpha_se["k"] + (1 - self.HT.se_per_contribution) * pe_contribution["k"]


            return mv_window, mq, mk_window

        def _step(self, input):

            # Reshape input into one more easily manageable for computing
            att_size = self.HT.context_size * self.HT.num_feat_patterns
            att_t_1_d = input[:att_size]
            att_t_1_d = anp.reshape(att_t_1_d, (self.HT.context_size, self.HT.num_feat_patterns))
            p_t_1_d = input[att_size:]
            p_t_1_d = anp.reshape(p_t_1_d, (self.HT.context_size, self.HT.pe_bit_size))

            # Compute mf
            mv_window, mq, mk_window = self.compute_mfs(att_t_1_d, p_t_1_d)

            # Compute A
            att_t_d = self.attention(att_t_1_d, mv_window, mq, mk_window)

            # Compute p
            p_t_d = self.PE.next_step_autograd(p_t_1_d)

            att_t_d = anp.reshape(att_t_d, att_size)
            p_t_d = anp.reshape(p_t_d, self.HT.context_size * self.HT.pe_bit_size)

            return anp.concatenate((att_t_d, p_t_d))


        def lyapunov_step(self, att_t_1_d, p_t_1_d, t, dx):

            Jacobian_Func = jacobian(self._step)

            att_t_1_d_flat = anp.reshape(att_t_1_d, self.HT.context_size * self.HT.num_feat_patterns)
            p_t_1_d_flat = anp.reshape(p_t_1_d, self.HT.context_size * self.HT.pe_bit_size)
            input = np.concatenate((att_t_1_d_flat, p_t_1_d_flat))

            # output = self._step(input)  # Output for testing
            J = Jacobian_Func(input)

            S_idx = t - self.HT.min_saved_step

            # Compute perturbation
            dx = np.matmul(J, dx)
            # Decompose perturbation
            Q, R = np.linalg.qr(dx)
            d_exp = np.absolute(np.diag(R))
            dS = np.log(d_exp)
            # Q is orthogonal so we can use it for the next step
            dx = Q

            self.S += dS
            self.S_i[S_idx] = dS
            self.S_i_sum[S_idx] = copy.deepcopy(self.S)

            return dx

        def lyapunov_end(self):

            self.S /= self.HT.num_saved_steps

            sorted_S = np.sort(self.S[:self.HT.num_feat_patterns * self.HT.context_size])[::-1]
            print("S", self.S)
            print("Sorted desc", sorted_S)
            print()

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(self.S_i_sum[:, 0:3])
            plt.title("Feats 0-2")
            plt.tight_layout()
            plt.show()
            plt.close()

            plt.figure()
            plt.plot(self.S_i_sum[-1000:, 0:3])
            plt.title("Feats 0-2. Zoom last steps")
            plt.tight_layout()
            plt.show()
            plt.close()

            plt.figure()
            plt.plot(self.S_i_sum[:, 3:12], )
            plt.title("Feats 3-11")
            plt.tight_layout()
            plt.show()
            plt.close()

            import pdb; pdb.set_trace()


    def set_beta_o(self, beta_o):
        self.beta_o = beta_o

    def set_beta_att(self, beta_att):
        self.beta_att = beta_att

    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def set_epsilon_pe(self, epsilon_pe):
        self.epsilon_pe = epsilon_pe

    def set_context_window(self, mv_window, mq_window, mk_window, att_window):
        self.mv_window = copy.deepcopy(mv_window)
        self.mq_window = copy.deepcopy(mq_window)
        self.mk_window = copy.deepcopy(mk_window)
        self.att = copy.deepcopy(att_window)
        self.att_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.att_window[0] = copy.deepcopy(att_window)


    def reset_data(self):

        self.mv_window = np.zeros((self.context_size, self.num_feat_patterns))
        # We don't need to order it every step normally. Order is only important for the Jacobian
        self.ordered_mv_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.mq_window = np.zeros(self.num_feat_patterns)
        self.mk_window = np.zeros((self.context_size, self.num_feat_patterns))
        # We don't need to order it every step normally. Order is only important for the Jacobian
        self.ordered_mk_window = np.zeros((self.context_size, self.num_feat_patterns))
        self.att = np.zeros(self.num_feat_patterns)
        self.att_window = np.zeros((self.context_size, self.num_feat_patterns))


        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, self.num_feat_patterns))

    def shift_d_window(self, shift):
        # Roll the context window by "shift" positions
        shifted_mv_window = np.roll(self.mv_window, shift, axis=0)
        shifted_mk_window = np.roll(self.mk_window, shift, axis=0)

        return shifted_mv_window, shifted_mk_window

    def reorder_context_window(self):
        # Shift values so the first value erased in the context window is the oldest one
        shift_amount = self.context_size - self.context_index - 1
        self.mv_window, self.mk_window = self.shift_d_window(shift_amount)

    def reset_data_keep_context(self):
        self.reorder_context_window()
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, self.num_feat_patterns))

    def qk_f_mf(self, tau):

        mqk = self.mq_window @ self.mk_window["mk"][tau]
        return self.beta_att * mqk

    @staticmethod
    def softmax(key_prob_unnorm):
        C = 10
        max_x = max(key_prob_unnorm)
        expp = np.exp(key_prob_unnorm - max_x + C)
        sum_exp = np.sum(expp)
        key_prob = expp / sum_exp

        return key_prob

    def der_att_dmv(self, key_prob_unnorm, effective_context_size):

        if self.run_exact_inf and \
                (self.normalize_weights_str_att == "N**2" or self.normalize_weights_str_att == "N**2*np.sqrt(M)"):

            key_prob_unnorm *= self.inf_normalization_att
            key_prob = self.softmax(key_prob_unnorm)

        else:
            raise Exception("\"normalize_weights_str_att\" is not either \"N**2\" or \"N**2*np.sqrt(M)\". "
                            "Please implement this method")

        datt_dmv_diag = effective_context_size * key_prob

        # We have the derivative of A_a wrt m^v_{b,d} for different d and a=c. When a!=c it's all zeros.
        # All the derivatives are equal for the different a's. It only changes wrt which d we are deriving to.

        # Create matrix of zeros in 3D (A_a, mv_c, context_size)
        datt_dmv = np.zeros((self.num_feat_patterns, self.num_feat_patterns, self.context_size))
        # Set only the diagonal elements, because when a!=c, all elements are zero
        diag_indices = np.diag_indices(self.num_feat_patterns, ndim=2)
        datt_dmv[diag_indices] = datt_dmv_diag

        return datt_dmv


    def der_att_dmq(self, key_prob_unnorm, effective_context_size):


        Z = np.sum(key_prob_unnorm)

        mk_exp = np.multiply(self.ordered_mk_window[:effective_context_size, :], key_prob_unnorm[:, np.newaxis])
        derv_part_1 = np.einsum("da,dc->ac", self.ordered_mv_window[:effective_context_size, :], mk_exp[:,:])
        derv_part_1 /= Z

        mv_exp = np.multiply(self.ordered_mv_window[:effective_context_size, :], key_prob_unnorm[:, np.newaxis])
        derv_part_2 = np.einsum("a,c->ac", np.sum(mv_exp, axis=0), np.sum(mk_exp, axis=0))
        derv_part_2 /= Z**2

        datt_dmq = self.gamma * (derv_part_1 - derv_part_2)

        # derv dimensions are (att_a, mq_c) we don't have a contex_size dimension because we don't keep copies of q
        # at different times

        return datt_dmq

    def der_att_dmk(self, key_prob_unnorm, effective_context_size):

        Z = np.sum(key_prob_unnorm)

        derv_term1 = self.ordered_mv_window[:effective_context_size, :] / Z
        mv_exp = np.multiply(self.ordered_mv_window[:effective_context_size, :], key_prob_unnorm[:, np.newaxis])
        derv_term2 = np.sum(mv_exp, axis=0) / Z ** 2

        derv_prod2 = derv_term1 - derv_term2

        mq_exp = np.einsum("c,u->uc",  self.gamma * self.mq_window, key_prob_unnorm)
        derv = np.einsum("dc,da->acd", mq_exp, derv_prod2)
        
        return derv  # Dimensions are (att_a, mk_c, mk_u)

    def der_att_dmq_loopy(self, key_prob_unnorm, effective_context_size):

        Z = np.sum(key_prob_unnorm)

        derv_part_1_loop = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
        derv_part_2_loop = np.zeros((self.num_feat_patterns, self.num_feat_patterns))

        for a in range(self.num_feat_patterns):
            for c in range(self.num_feat_patterns):
                prod1 = 0
                prod2 = 0
                for d in range(effective_context_size):
                    derv_part_1_loop[a, c] += (self.ordered_mv_window[d, a] * self.ordered_mk_window[d, c]
                                             * key_prob_unnorm[d])

                    prod1 += (self.ordered_mv_window[d, a] * key_prob_unnorm[d])
                    prod2 += (self.ordered_mk_window[d, c] * key_prob_unnorm[d])
                    derv_part_2_loop[a, c] = prod1 * prod2
        derv_part_1_loop /= Z
        derv_part_2_loop /= Z**2

        derv_loop = self.gamma * (derv_part_1_loop - derv_part_2_loop)

        return derv_loop  # Dimensions are (att_a, mq_c)

    def der_att_dmk_loopy(self, key_prob_unnorm, effective_context_size):

        Z = np.sum(key_prob_unnorm)

        derv_loopy = np.zeros((self.num_feat_patterns, self.num_feat_patterns, effective_context_size))

        for a in range(self.num_feat_patterns):
            term2_a = 0
            for d in range(effective_context_size):
                term2_a += (self.ordered_mv_window[d, a] * key_prob_unnorm[d])
            term2_a /= Z**2

            for c in range(self.num_feat_patterns):
                for u in range(effective_context_size):
                    derv_loopy[a, c, u] = (self.gamma * self.mq_window[c] * key_prob_unnorm[u] *
                                           (self.ordered_mv_window[u, a] / Z - term2_a))

        return derv_loopy  # Dimensions are (att_a, mk_c, mk_u)


    def attention_derivatives(self, t):
        effective_context_size = min(self.context_size, t + 1)


        if (not self.run_exact_inf and
                (self.normalize_weights_str_att == "N**2" or self.normalize_weights_str_att == "N**2*np.sqrt(M)")):

            raise Exception(f"Scaling is not well defined for the derivatives for "
                            f"normalize_weights_str_att={self.normalize_weights_str_att}")

        dAdmv = self.der_att_dmv(self.key_prob_unnorm, effective_context_size)
        dAdmq = self.der_att_dmq(self.key_prob_unnorm, effective_context_size)
        dAdmk = self.der_att_dmk(self.key_prob_unnorm, effective_context_size)

        return dAdmv, dAdmq, dAdmk

    def f_dm_dA(self, att_1):
        """
        Derivatives of the mean-fields wrt. attention
        :return:
        """

        if self.run_exact_inf:  # In infty, we are going to deal with the order of N in the output.

            sign_att_patterns = (self.beta_o * self.scaling_o *
                                 np.einsum("jb,b->j", self.sign_matrix[:, :self.num_feat_patterns],
                                           att_1))

            idx_not_zero = np.where(sign_att_patterns != 0)
            # If result is not 0, normalize by inf (avoids NaNs)
            sign_att_patterns[idx_not_zero] *= self.inf_normalization_o
            d_tanh_j_signs = 1 - np.tanh(sign_att_patterns)**2
        else:  # Otherwise, handle it here, just compute tanh
            d_tanh_j_signs = 1 - np.tanh(self.beta_o * self.scaling_o * (1 / self.normalizing_constant_o) *
                                   np.einsum("jb,b->j", self.sign_matrix[:, :self.num_feat_patterns],
                                             att_1))**2

        d_tanh_j_signs_c = np.einsum("j,jc->jc", d_tanh_j_signs, self.sign_matrix[:, :self.num_feat_patterns])

        # Compute the semantic part of every mean field needed for the attention
        dm_alpha_se_dA = {}
        for feat_name in ["v", "q", "k"]:
            # Index j is the different combinations of sigma patterns
            dm_alpha_se_dA[feat_name] = (np.einsum("ja,jc->ac", self.corr_signed[feat_name],
                                                self.beta_o * self.scaling_o * d_tanh_j_signs_c))  # Size (num_features)

            dm_alpha_se_dA[feat_name] = self.se_per_contribution * dm_alpha_se_dA[feat_name] / 2 ** (self.num_feat_patterns - 1)

        return dm_alpha_se_dA

        # Loopy implementation for testing
        # dm_alpha_se_loop = {}
        # for feat_name in self.features_names:
        #     dm_alpha_se_loop[feat_name] = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
        #     for a in range(self.num_feat_patterns):
        #         for c in range(self.num_feat_patterns):
        #             for j in range(len(self.corr_signed[feat_name])):
        #                 var1 = 0
        #                 var2 = 0
        #                 for b in range(self.num_feat_patterns + 1):
        #                     var1 += self.sign_matrix[j,b] * self.even_corr[feat_name][b,a]
        #                 for b in range(self.num_feat_patterns):
        #                     var2 += self.sign_matrix[j,b] * att[b]
        #
        #                 var2 = var2 *  self.beta_o * self.scaling_o
        #                 tanh_var = 1 - np.tanh(var2)**2
        #                 dm_alpha_se_loop[feat_name][a,c] += (var1 * tanh_var * self.beta_o * self.scaling_o
        #                                                     * self.sign_matrix[j,c])
        #
        #     dm_alpha_se_loop[feat_name] = self.se_per_contribution * dm_alpha_se_loop[feat_name] / 2 ** (self.num_feat_patterns - 1)


    def f_dA_dA_or_P(self, t, dm_alpha_se_dA_or_P):

        effective_context_size = min(self.context_size, t + 1)

        # self.key_prob_unnorm equals G in the latex notation

        # local t, where t index is found within the context window
        lt = self.context_index

        Z = np.sum(self.key_prob_unnorm)

        if (not self.run_exact_inf and
                (self.normalize_weights_str_att == "N**2" or self.normalize_weights_str_att == "N**2*np.sqrt(M)")):
            raise Exception(f"Scaling is not well defined for the derivatives for "
                            f"normalize_weights_str_att={self.normalize_weights_str_att}")

        # First term:
        # First term within the first term
        dmv_dA_key_prob = np.einsum("ac,t -> tac", dm_alpha_se_dA_or_P["v"], self.key_prob_unnorm)

        # Second term within the first term
        dmq_dA_mk = np.einsum('bc,tb -> tc', dm_alpha_se_dA_or_P["q"],
                           self.mk_window[:effective_context_size], optimize=True)

        mv_key_prob = np.einsum('ta,t -> ta', self.mv_window[:effective_context_size],
                                self.key_prob_unnorm, optimize=True)

        mv_key_prob_dmq_mk = (np.einsum('ta,tc -> tac', mv_key_prob, dmq_dA_mk, optimize=True)
                              * self.gamma)

        # Third term within the first term
        mq_dmk_dA = np.einsum('b,bc -> c', self.mq_window,
                              dm_alpha_se_dA_or_P["k"], optimize=True)
        # mv_key_prob_t of size a
        mv_key_prob_t = self.mv_window[lt] * self.key_prob_unnorm[lt]
        mv_key_prob_t_mq_dmk = (np.einsum('a,c -> ac', mv_key_prob_t, mq_dmk_dA, optimize=True)
                              * self.gamma)

        # Aggregate the first term
        first_term = dmv_dA_key_prob + mv_key_prob_dmq_mk
        # Sum over tau dimension (0)
        first_term = np.sum(first_term, axis=0)
        first_term += mv_key_prob_t_mq_dmk
        first_term /= Z

        # Second term:
        # First term within the second term
        mv_key_prob = np.einsum("ta,t -> a", self.mv_window, self.key_prob_unnorm)
        # Second term within the second term
        key_prob_dmq_mk = np.einsum('t,tc -> c', self.key_prob_unnorm, dmq_dA_mk, optimize=True)
        # Third term within the second term
        key_prob_t_mq_dmk = self.key_prob_unnorm[lt] * mq_dmk_dA

        second_term = self.gamma * (key_prob_dmq_mk + key_prob_t_mq_dmk)

        second_term = np.einsum('a,c -> ac', mv_key_prob, second_term)
        second_term /= Z**2

        dA_dA = first_term - second_term

        #
        # # Loopy implementation for testing
        # c_size = dm_alpha_se_dA_or_P["q"].shape[1]
        #
        # term_1_loopy_test = np.zeros((self.num_feat_patterns, c_size))
        # term_2_loopy_test = np.zeros((self.num_feat_patterns, c_size))
        #
        # # a is the variable we are deriving
        # for a in range(self.num_feat_patterns):
        #
        #     # First term within the second term
        #     numerator_term2_part1_test = 0
        #     for tau in range(effective_context_size):
        #         numerator_term2_part1_test += self.mv_window[tau, a] * self.key_prob_unnorm[tau]
        #
        #     # c is the variable wrt we are deriving
        #     for c in range(c_size):
        #         numerator_term1_test = 0
        #         numerator_term2_part2_test = 0
        #
        #         for tau in range(effective_context_size):
        #             # First term within the first term
        #             numerator_term1_test += dm_alpha_se_dA_or_P["v"][a, c] * self.key_prob_unnorm[tau]
        #
        #             # Part of the second term of both terms
        #             dmq_dA_mk_loop_test = 0
        #             for b in range(self.num_feat_patterns):
        #                 dmq_dA_mk_loop_test += dm_alpha_se_dA_or_P["q"][b, c] * self.mk_window[tau, b]
        #
        #             second_term_both_terms = self.key_prob_unnorm[tau] * self.gamma * dmq_dA_mk_loop_test
        #             numerator_term1_test += self.mv_window[tau, a] * second_term_both_terms
        #             numerator_term2_part2_test += second_term_both_terms
        #
        #         mq_dmk_dA_loop_test = 0
        #         for b in range(self.num_feat_patterns):
        #             mq_dmk_dA_loop_test += self.mq_window[b] * dm_alpha_se_dA_or_P["k"][b, c]
        #
        #         third_term_both_terms =  self.key_prob_unnorm[lt] * self.gamma * mq_dmk_dA_loop_test
        #         numerator_term1_test += self.mv_window[lt, a] * third_term_both_terms
        #         numerator_term2_part2_test += third_term_both_terms
        #
        #         term_1_loopy_test[a, c] = numerator_term1_test
        #         term_2_loopy_test[a, c] = numerator_term2_part1_test * numerator_term2_part2_test
        #
        # term_1_loopy_test /= Z
        # term_2_loopy_test /= Z**2
        # dA_dA_loopy_test = term_1_loopy_test - term_2_loopy_test
        #
        # if not np.allclose(dA_dA, dA_dA_loopy_test):
        #     print("Review", t)

        return dA_dA


    def initialize_jacobian(self):
        jacobian_size = (self.num_feat_patterns + self.pe_bit_size)
        self.J = np.zeros((jacobian_size, jacobian_size))
        # der_m_der_p does not change though time
        self.dm_dp = self.f_dm_dp()


    def jacobian(self, t, att_1):
        # Dictionary with the derivatives of m wrt A for "v" "q" and "k"
        dm_alpha_se_dA = self.f_dm_dA(att_1)

        dA_dA = self.f_dA_dA_or_P(t, dm_alpha_se_dA)
        dA_dP = self.f_dA_dA_or_P(t, self.dm_dp)

        self.J[:self.num_feat_patterns, :self.num_feat_patterns] = dA_dA
        self.J[:self.num_feat_patterns, self.num_feat_patterns:] = dA_dP
        self.J[self.num_feat_patterns:, self.num_feat_patterns:] = self.PE.dp_dp

        print(np.linalg.det(self.J))

        self.J_p = self.PE.dp_dp

    def compute_lyapunov(self, t, dx, dx_p):

        S_idx = t - self.min_saved_step

        self.jacobian(t, self.att_window)

        # Compute perturbation
        dx = np.matmul(self.J, dx)
        # Decompose perturbation
        Q, R = np.linalg.qr(dx)
        d_exp = np.absolute(np.diag(R))
        dS = np.log(d_exp)
        # Q is orthogonal so we can use it for the next step
        dx = Q

        dx_p = np.matmul(self.J_p, dx_p)
        Q_p, R_p = np.linalg.qr(dx_p)
        d_exp_P = np.absolute(np.diag(R_p))
        dS_p = np.log(d_exp_P)
        dx_p = Q_p

        self.S_p += dS_p
        self.S_p_i[S_idx] = dS_p
        self.S_p_i_sum[S_idx] += dS_p

        self.S += dS
        self.S_i[S_idx] = dS
        self.S_i_sum[S_idx] = copy.deepcopy(self.S)

        return dx, dx_p



    def f_dm_dm(self, att, dAdm):
        if self.run_exact_inf:  # In infty, we are going to deal with the order of N in the output.

            sign_att_patterns = (self.beta_o * self.scaling_o *
                                 np.einsum("jb,b->j", self.sign_matrix[:, :self.num_feat_patterns],
                                           att))

            idx_not_zero = np.where(sign_att_patterns != 0)
            # If result is not 0, normalize by inf (avoids NaNs)
            sign_att_patterns[idx_not_zero] *= self.inf_normalization_o
            d_tanh_j_signs = 1 - np.tanh(sign_att_patterns)**2
        else:  # Otherwise, handle it here, just compute tanh
            d_tanh_j_signs = 1 - np.tanh(self.beta_o * self.scaling_o * (1 / self.normalizing_constant_o) *
                                   np.einsum("jb,b->j", self.sign_matrix[:, :self.num_feat_patterns],
                                             att))**2

        # Compute the semantic part of every mean field needed for the attention
        dm_alpha_se = {}
        for feat_name in self.features_names:
            dm_alpha_se[feat_name] = (self.se_per_contribution * np.einsum("ja,j->ja",
                                                                          self.corr_signed[feat_name], d_tanh_j_signs)
                                     / 2 ** (self.num_feat_patterns - 1)) # Size (num_features)

        sign_dAtt_patterns_dm = {}
        for feat_name in self.features_names:
            if feat_name == "o":  # We only have the derivatives wrt v q k
                continue
            elif feat_name == "q":
                sign_dAtt_patterns_dm[feat_name] = (self.beta_o * self.scaling_o *  self.inf_normalization_o *
                                                    np.einsum("jb,bc->jbc",
                                                              self.sign_matrix[:, :self.num_feat_patterns],
                                                              dAdm[feat_name]))
            else:
                sign_dAtt_patterns_dm[feat_name] = (self.beta_o * self.scaling_o *  self.inf_normalization_o *
                                 np.einsum("jb,bcd->jbcd", self.sign_matrix[:, :self.num_feat_patterns],
                                           dAdm[feat_name]))

        # Compute the derivatives of the mean-fields wrt to each other
        dm_dm = {}
        feats = ["v", "q", "k"]
        for feat_name_1 in feats:
            dm_dm[feat_name_1] = {}
            for feat_name_2 in feats:
                if feat_name_2 == "q":
                    dm_dm[feat_name_1][feat_name_2] = (
                        np.einsum("ja,jbc->ac", dm_alpha_se[feat_name_1],
                                  sign_dAtt_patterns_dm[feat_name_2]))
                else:  # v or k
                    dm_dm[feat_name_1][feat_name_2] = (
                        np.einsum("ja,jbcd->acd", dm_alpha_se[feat_name_1],
                                  sign_dAtt_patterns_dm[feat_name_2]))


        return dm_dm

    def f_dm_dp(self):
        dm_dp  = {}
        for feat in self.features_names:
            dm_dp[feat] = (1 - self.se_per_contribution) * self.W_dict[feat][:, -self.pe_bit_size:] / self.pe_bit_size

        return dm_dp

    def initialize_degenerate_jacobian(self):
        # Initialize jacobian, dims
        # (num_mean_field_types x num_features x context_size + num_bits_pe)
        # order mean-fields: o, v, q, k
        # order context_size: 0, 1, 2... u=0, current time, u>0 delay
        # order features: 0, 1, 2, ...
        # cols: m^v_{a,t-1,d=0}, m^v_{a,t-1,d=1}, ..., m^v_{a,t-1,d=2}, m^v_{b,t-1,d=0}, ... m^q_{a,t-1,d=0}, ...
        # cols: m^v_{a,t-1,d=0}, m^v_{b,t-1,d=0}, ..., m^v_{c,t-1,d=0}, m^v_{a,t-1,d=1}, ... m^q_{a,t-1,d=0}, ...

        # v and k have num_feat_patterns*context_size,q only has num_feat_patterns
        # We have "context_size" copies of the position that we rotate every timestep
        jacobian_size = (2 * self.num_feat_patterns * self.context_size + self.num_feat_patterns +
                             self.pe_bit_size * self.context_size)
        self.J = np.zeros((jacobian_size, jacobian_size))

        dm_dp = self.f_dm_dp()

        # The derivatives of mean-fields of d>0 wrt d-1 are constant and equal to 1 for large simulations.

        # Idxs of start of v, q, k, pe sections in the matrix
        self.J_mf_type_start_idxs = [0, 0, 0, 0, None]
        # q start
        self.J_mf_type_start_idxs[1] = self.J_mf_type_start_idxs[0] + self.num_feat_patterns * self.context_size
        # k start
        self.J_mf_type_start_idxs[2] = self.J_mf_type_start_idxs[1] + self.num_feat_patterns
        # pe start
        self.J_mf_type_start_idxs[3] = self.J_mf_type_start_idxs[2] + self.num_feat_patterns * self.context_size

        I = np.identity(self.num_feat_patterns * (self.context_size-1))

        # for v and k, set the derivatives of m^alpha_{a,d>0} wrt m^alpha_{a,d-1} = 1
        # In the way the jacobian is ordered, this is accomplished by setting the diagonal elements of
        # \frac{\partial m^{\alpha}_{t+1,d}}{\partial m^{\alpha}_{t,u}} for d>1
        derivative_feat_names = ["v", "q", "k"]
        first_loop = zip(list(range(3)), derivative_feat_names)
        for i, feat_1 in first_loop:
            # if feat_1 == "o" or feat_1 == "q": continue

            # Indices of the derivatives of mean-fields with d=0
            idx_i_0 = self.J_mf_type_start_idxs[i]
            idx_i_1 = self.J_mf_type_start_idxs[i+1]


            # Treat in this inner loop the derivatives of mean-fields wrt mean-fields
            second_loop = zip(list(range(3)), derivative_feat_names)
            for j, feat_2 in second_loop:
                if feat_2 == "o" or feat_2 == "q": continue
                if feat_1 == feat_2:
                    idx_j_0 = self.J_mf_type_start_idxs[j]
                    idx_j_1 = self.J_mf_type_start_idxs[j + 1]

                    # Start and end indices for setting the diagonal to 1. (That is, to set the derivatives wrt to copies).
                    idx_i_0_diag = idx_i_0 + self.num_feat_patterns
                    idx_j_1_diag = idx_j_1 - self.num_feat_patterns

                    self.J[idx_i_0_diag:idx_i_1, idx_j_0:idx_j_1_diag] = I

            # Treat now the derivatives of mean-fields wrt positional encodings
            dalpha_dp = dm_dp[feat_1]
            idx_i_1_mf_pe = idx_i_0 + self.num_feat_patterns
            idx_j_0_pe = self.J_mf_type_start_idxs[-2]
            idx_j_1_pe = idx_j_0_pe + self.pe_bit_size
            self.J[idx_i_0:idx_i_1_mf_pe, idx_j_0_pe:idx_j_1_pe] = dalpha_dp

        # Now treat the derivatives of the positional encoding with respect to itself
        # p_{i,t+1,d} = p_{i,t,d-1 mod context_size}
        idx_i_0_pe = self.J_mf_type_start_idxs[-2]
        idx_i_1_pe = self.J_mf_type_start_idxs[-1]
        idx_j_0_pe = self.J_mf_type_start_idxs[-2]
        # First, p_{i,t+1,0} = p_{i,t,CS-1}
        I = np.identity(self.pe_bit_size)
        self.J[idx_i_0_pe:idx_i_0_pe+self.pe_bit_size, -self.pe_bit_size:] = I
        # Then set the remaining
        I = np.identity(self.pe_bit_size * (self.context_size - 1))
        self.J[idx_i_0_pe+self.pe_bit_size:idx_i_1_pe, idx_j_0_pe:-self.pe_bit_size] = I

    def degenerate_jacobian(self, t, att):

        shift_amount = self.context_size - self.context_index - 1
        self.ordered_mv_window, self.ordered_mk_window = self.shift_d_window(shift_amount)

        # Compute the derivatives of the attention wrt MFs v q k
        dAdmv, dAdmq, dAdmk = self.attention_derivatives(t)
        # Create dict structure
        dAdm = {"v": dAdmv, "q": dAdmq, "k": dAdmk}
        dm_dm = self.f_dm_dm(att, dAdm)

        derivative_feat_names = ["v", "q", "k"]
        first_loop = zip(list(range(3)), derivative_feat_names)
        for i, feat_1_name in first_loop:
            # Indices of the derivatives of mean-fields with d=0
            idx_i_0 = self.J_mf_type_start_idxs[i]
            idx_i_1 = idx_i_0 + self.num_feat_patterns

            second_loop = zip(list(range(3)), derivative_feat_names)
            for j, feat_2_name in second_loop:

                idx_j_0 = self.J_mf_type_start_idxs[j]
                idx_j_1 = self.J_mf_type_start_idxs[j + 1]
                num_cols = idx_j_1 - idx_j_0

                dmalpha_dmlambda_flatten = dm_dm[feat_1_name][feat_2_name]
                # Reshape features with copies
                if feat_2_name == "v" or feat_2_name == "k":
                    dmalpha_dmlambda_flatten = np.reshape(dm_dm[feat_1_name][feat_2_name],
                                                      (self.num_feat_patterns, num_cols), order="F")

                self.J[idx_i_0:idx_i_1, idx_j_0:idx_j_1] = dmalpha_dmlambda_flatten


    def key_averaging(self, key_prob_unnorm, effective_context_size):

        if self.run_exact_inf:

            if self.normalize_weights_str_att == "N**2" or self.normalize_weights_str_att == "N**2*np.sqrt(M)":
                key_prob = self.softmax(key_prob_unnorm)

                # We'll deal with normalization in the mf_computation function, but since we are returning
                # the average of mvs and not N*mv, we are already normalizing by a /N factor
                self.att = np.einsum("da,d->a", self.mv_window[:effective_context_size], key_prob)

            elif self.normalize_weights_str_att != "N**2":
                # In infty the softmax saturates and evolves into argmax
                max_val = max(key_prob_unnorm)
                max_ids = np.argwhere(key_prob_unnorm == max_val)
                selected_mvs = self.mv_window[max_ids]

                # The array created has 1 more empty dimension than we need, so we index by 0
                self.att = np.mean(selected_mvs, axis=0)[0]
                # We'll deal with normalization in the mf_computation function, but since we are returning
                # the average of mvs and not N*mv, we are already normalizing by a /N factor
        else:
            key_prob = self.softmax(self.N_normalization ** 2 * key_prob_unnorm / self.normalizing_constant_att)
            self.att = (self.N_normalization *
                        np.einsum("da,d->a", self.mv_window[:effective_context_size], key_prob))

    # def attention_unoptimized(self, t):
    #
    #     effective_context_size = min(self.context_size, t + 1)
    #
    #     key_prob_unnorm = np.zeros(effective_context_size)
    #     for tau in range(0, effective_context_size):
    #         key_prob_unnorm[tau] = self.qk_f_mf(tau)
    #
    #     # Cannot be used for testing anymore, it's missing the scaling and normalizing factors here
    #
    #     self.key_averaging(key_prob_unnorm, effective_context_size)
    #
    #     # # Loopy implementation for testing
    #     #
    #     # att_t_loopy = np.zeros(self.num_feat_patterns)
    #     # for b in range(0, self.num_feat_patterns):
    #     #     for tau in range(0, t+1):
    #     #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]
    #
    #     if t >= self.min_saved_step:
    #         self.mf_statistics["att"][t - self.min_saved_step] = copy.deepcopy(self.att_window)

    def attention(self, t):

        effective_context_size = min(self.context_size, t + 1)
        # Put in common queries and keys
        mqk = np.einsum('b,tb -> t', self.mq_window, self.mk_window[:effective_context_size],
                        optimize=True)

        # Scale
        if self.run_exact_inf:
            self.key_prob_unnorm = self.gamma * mqk
        else:
            self.key_prob_unnorm = self.beta_att * self.scaling_att * mqk


        # Compute softmax and average by mv
        self.key_averaging(self.key_prob_unnorm, effective_context_size)

        if self.compute_jacobian:
            # Maintain the attention window for the jacobian computation
            self.att_window = np.roll(self.att_window, 1, axis=0)
            self.att_window[0] = copy.deepcopy(self.att)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        if t >= self.min_saved_step:  # Save if required
            self.mf_statistics["att"][t - self.min_saved_step] = copy.deepcopy(self.att)

    def save_stats(self, t, mo, mo_se, mv, mq, mk):
        index_t = t - self.min_saved_step
        self.mf_statistics["mo"][index_t] = copy.deepcopy(mo)
        self.mf_statistics["mo_se"][index_t] = copy.deepcopy(mo_se)
        self.mf_statistics["mv"][index_t] = copy.deepcopy(mv)
        self.mf_statistics["mq"][index_t] = copy.deepcopy(mq)
        self.mf_statistics["mk"][index_t] = copy.deepcopy(mk)

    def compute_means_from_data(self, x0, t, ponder_pe=True):
        # In the first version of the paper ponder_pe=False for the bif diagrams


        # Computes mean values from data. Basically used to compute m values from a x0 token.
        self.context_index = t % self.context_size

        if ponder_pe:

            self.mv_window[self.context_index, :] = \
                ((self.se_per_contribution * x0[:self.se_bit_size] @ self.W_dict["v"][:, :self.se_bit_size].T / self.se_bit_size) +
                 ((1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["v"][:, -self.pe_bit_size:].T / self.pe_bit_size))

            self.mq_window = \
                ((self.se_per_contribution * x0[:self.se_bit_size] @ self.W_dict["q"][:, :self.se_bit_size].T / self.se_bit_size) +
                 ((1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["q"][:, -self.pe_bit_size:].T / self.pe_bit_size))

            self.mk_window[self.context_index, :] = \
                ((self.se_per_contribution * x0[:self.se_bit_size] @ self.W_dict["k"][:, :self.se_bit_size].T / self.se_bit_size) +
                 ((1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["k"][:, -self.pe_bit_size:].T / self.pe_bit_size))

            if t >= self.min_saved_step:
                mo_se =  x0[:self.se_bit_size] @ self.W_dict["o"][:, :self.se_bit_size].T / self.se_bit_size

                mo = (self.se_per_contribution * mo_se +
                      (1 - self.se_per_contribution) * x0[-self.pe_bit_size:] @ self.W_dict["o"][:, -self.pe_bit_size:].T / self.pe_bit_size)

                self.save_stats(t, mo, mo_se, self.mv_window[self.context_index, :], self.mq_window,
                                self.mk_window[self.context_index, :])

        else:
            self.mv_window[self.context_index, :] = x0 @ self.W_dict["v"].T / self.embedding_size
            self.mq_window = x0 @ self.W_dict["q"].T / self.embedding_size
            self.mk_window[self.context_index, :] = x0 @ self.W_dict["k"].T / self.embedding_size

            if t >= self.min_saved_step:
                mo = x0 @ self.W_dict["o"].T / self.embedding_size
                mo_se = x0[:self.se_bit_size] @ self.W_dict["o"][:, :self.se_bit_size].T / self.se_bit_size

                self.save_stats(t, mo, mo_se, self.mv_window[self.context_index, :], self.mq_window,
                                self.mk_window[self.context_index, :])

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

    def compute_mf(self, t):
        # Load attention values
        att = self.att
        # Encode the position
        pos_vec = self.PE.getter()

        # Positional Embedding contribution
        pe_contribution = {}
        for feat_name in self.features_names:
            pe_contribution[feat_name] = np.einsum('bi,i ->b', self.W_dict[feat_name][:, self.se_bit_size:],
                                      pos_vec, optimize=True) / self.pe_bit_size

        if self.run_exact_inf:  # In infty, we are going to deal with the order of N in the output.

            sign_att_patterns = (self.beta_o * self.scaling_o *
                                 np.einsum("jb,b->j", self.sign_matrix[:, :self.num_feat_patterns],
                                           att))

            idx_not_zero = np.where(sign_att_patterns != 0)
            # If result is not 0, normalize by inf (avoids NaNs)
            sign_att_patterns[idx_not_zero] *= self.inf_normalization_o
            tanh_j_signs = np.tanh(sign_att_patterns)
        else:  # Otherwise, handle it here, just compute tanh
            tanh_j_signs = np.tanh(self.beta_o * self.scaling_o * (1 / self.normalizing_constant_o) *
                                   np.einsum("jb,b->j", self.sign_matrix[:, :self.num_feat_patterns],
                                             att))

        # Compute relative context index
        self.context_index = t % self.context_size

        # Compute the semantic part of every mean field needed for the attention
        m_alpha_se = {}
        for feat_name in self.features_names:
            # corr_signed has shape (num_combinations_signs, num_features_a).
            # For every feature a, you put together all the j combinations of signs
            m_alpha_se[feat_name] = (self.se_per_contribution * np.einsum("ja,j->a",
                                                                          self.corr_signed[feat_name], tanh_j_signs)
                                 / 2 ** (self.num_feat_patterns - 1))

        self.mv_window[self.context_index] = m_alpha_se["v"] + (1 - self.se_per_contribution) * pe_contribution["v"]
        self.mq_window = m_alpha_se["q"] + (1 - self.se_per_contribution) * pe_contribution["q"]
        self.mk_window[self.context_index] = m_alpha_se["k"] + (1 - self.se_per_contribution) * pe_contribution["k"]



        # Compute mean-field for o. Separate the behavior of the Semantic Embedding.
        if t >= self.min_saved_step:
            m_alpha_se["o"] = (np.einsum("jb,j->b", self.corr_signed["o"], tanh_j_signs)
                     / 2 ** (self.num_feat_patterns - 1))
            mo = (self.se_per_contribution * m_alpha_se["o"] + (1 - self.se_per_contribution) * pe_contribution["o"])

            self.save_stats(t, mo, m_alpha_se["o"], self.mv_window[self.context_index, :], self.mq_window,
                            self.mk_window[self.context_index, :])


    def return_context_window(self):
        return self.att, self.mv_window, self.mq_window, self.mk_window

    def simulate_mf_from_context(self, max_steps, compute_lyapunov=True):
        # In order for this method to work properly, a simulate_mf() method has had to be run previously at least for
        # self.context_size steps

        # We have in self.att_window the last attention value
        # We initialize the model at the end of the previous execution
        # The context window has been reordered before saving so the last element is in the last position
        ini_t = self.context_size
        self.PE.initialize_state(0)

        if compute_lyapunov:
            # Initialize Jacobian if needed
            self.initialize_jacobian()
            dx = np.eye(self.num_feat_patterns * self.context_size + self.pe_bit_size * self.context_size)

        for t in range(ini_t, max_steps):

            self.t = t
            if compute_lyapunov and (t >= self.min_saved_step):
                dx = self.L.lyapunov_step(self.att_window, self.PE.state_window, t, dx)


            self.compute_mf(t)
            self.attention(t)

            # if compute_lyapunov and (t >= self.min_saved_step):
            #     dx, dx_p = self.compute_lyapunov(t, dx, dx_p)

            self.PE.next_step()


        if compute_lyapunov:
            self.L.lyapunov_end()



    def simulate(self, x0, max_steps, compute_lyapunov=True):

        self.PE.initialize_state(0)

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(x0, t=0)
        self.attention(t=0)

        self.initialize_jacobian()

        for t in range(1, max_steps):
            self.t = t
            self.compute_mf(t)
            self.attention(t)

            # if compute_lyapunov and (t >= self.min_saved_step):
            #     dx, dx_p = self.compute_lyapunov(t, dx, dx_p)

            self.PE.next_step()



        if compute_lyapunov:
            self.L.S /= self.num_saved_steps

            sorted_S = np.sort(self.S)[::-1]

