import copy
import numpy as np
from autograd import numpy as anp
from autograd import jacobian
from utils import signed_binary_encoding
from models.PositionalEncoding import PositionalEncoding

from models.sigma_configs import (set_orders_manually_3_no_duplicity, set_orders_manually_4_duplicity_loop_1,
                                   set_orders_manually_4_duplicity_loop_2, set_orders_manually_4_duplicity_loop_3,
                                  set_orders_manually_4_duplicity_loop_no_pe, set_orders_manually_4_no_duplicity,
                                   set_orders_manually_6_duplicity_loop, set_orders_manually_4_cycle_0_1_2_1)


class HopfieldSelfAttentionNNMFInfNPESigma:

    def __init__(self, beta_o, gamma_att, num_feat_patterns, num_feat_patterns_se,
                 positional_embedding_bitsize, vocab, context_size,
                 max_sim_steps=512, min_saved_step=0,
                 epsilon_pe=0.95,
                 seed_W=0,
                 jacobian=True):


        # Config
        self.num_feat_patterns = num_feat_patterns # Number of mean-field patterns > num_feat_patterns_se
        self.num_feat_patterns_se = num_feat_patterns_se # Number of original mean-field patterns
        self.pe_bit_size = positional_embedding_bitsize
        self.context_size = context_size
        self.max_sim_steps = max_sim_steps
        self.min_saved_step = min_saved_step
        self.num_saved_steps = max_sim_steps - min_saved_step

        self.seed_W = seed_W

        if num_feat_patterns < num_feat_patterns_se:
            raise ValueError("num_feat_patterns must be >= num_feat_patterns_se")

        # `se_per_contribution` must be defined like this for reproducibility of the paper results
        # Otherwise results differ a little for changes in small decimals
        self.epsilon_pe = epsilon_pe
        self.epsilon_se = 1 - epsilon_pe
        self.beta_o = beta_o
        self.gamma = gamma_att

        # Set up Positional Encoding
        self.PE = PositionalEncoding(positional_embedding_bitsize, vocab, self.context_size, K=10, type="base")


        # Set-up features config, correlations and weights
        self.features_names = ["q", "k", "v", "o", ]
        self.feature_to_idx = {name: i for i, name in enumerate(self.features_names)}

        self.create_correlations_manually()

        # Create variables to save results
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        # Create variables for saving the statistics of the mean-field model
        self.mf_statistics = {}
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_steps, num_feat_patterns))
        self.mf_statistics["m_tilde"] = np.zeros((self.num_saved_steps, 4, num_feat_patterns_se))
        self.mf_statistics["m_tilde_proj"] = np.zeros((self.num_saved_steps, 4, num_feat_patterns))
        self.mf_statistics["m_pos"] = np.zeros((self.num_saved_steps, 4, num_feat_patterns))

        self.compute_jacobian = jacobian

        # Variables for accumulating the Lyapunov exponents and debugging
        self.lyapunov_size = self.num_feat_patterns * self.context_size + self.pe_bit_size * self.context_size
        self.S = np.zeros(self.lyapunov_size)
        self.S_i = np.zeros((self.num_saved_steps, self.lyapunov_size))
        self.S_i_sum = np.zeros((self.num_saved_steps, self.lyapunov_size))
        self.S_inf_flag = np.zeros(self.lyapunov_size)


    def build_B_P_from_orders(self, orders, num_base_patterns):
        """
        orders:

        For B (C, M) with values in [0, M-1], it will be a permutation
        For P (C, K) with values in [0, M-1], orders will have repeated indexes

        returns:
            B: (C, M, M)
            or
            P: (C, K, M)
            depending on the use case
        """
        C, K = orders.shape

        out = np.zeros((C, K, num_base_patterns), dtype=int)

        rows = np.arange(K)[None, :]        # (1, M)
        feats = np.arange(C)[:, None]       # (F, 1)

        out[feats, rows, orders] = 1

        return out


    def create_correlations_manually(self, ini_m_idx=0):

        self.orders_B, orders_P, self.W = set_orders_manually_4_duplicity_loop_3(self, ini_m_idx=ini_m_idx)

        # Create the B_alpha matrices to indicate the order of the mean-field patterns for the vectorized computation of the mean-field values
        self.B_alpha = self.build_B_P_from_orders(self.orders_B, self.num_feat_patterns)

        # Pre-compute sigma
        self.sigma = signed_binary_encoding(np.arange(2 ** self.num_feat_patterns_se), self.num_feat_patterns_se)  # (2^M, M)

        # (C, K, M)
        self.P_alpha = self.build_B_P_from_orders(orders_P, self.num_feat_patterns_se)

        # ---- 2. project all sigma ----
        # sigma: (2^M, M)
        # P: (C, K, M)
        # (C, 2^M, K)
        self.sigma_proj_alpha = np.einsum('nm,ckm->cnk', self.sigma, self.P_alpha)

    def reconfigure_pe_for_ini_m_idx(self, ini_m_idx):
        """
        Reconfigure the PE (positional encoding) based on the initial pattern index.
        This ensures the cycle dynamics are properly aligned with the starting position.
        
        Call this before simulate() if ini_m_idx changes.
        """
        self.create_correlations_manually(ini_m_idx=ini_m_idx)


    def set_beta_o(self, beta_o):
        self.beta_o = beta_o

    def set_gamma_att(self, gamma_att):
        self.gamma_att = gamma_att

    def set_betas(self, beta_o, gamma_att):
        self.beta_o = beta_o
        self.gamma_att = gamma_att

    def set_epsilon_pe(self, epsilon_pe):
        self.epsilon_pe = epsilon_pe
        self.epsilon_se = 1 - epsilon_pe


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


    def save_mf_stats(self, mo, mo_se, mv, mq, mk, m_tilde, m_tilde_proj, m_pos):
        # Save stats in an array if the time threshold is surpassed
        if self.t >= self.min_saved_step:

            index_t = self.t - self.min_saved_step
            if isinstance(mv, np.ndarray):
                self.mf_statistics["mo"][index_t] = copy.deepcopy(mo)
                self.mf_statistics["mo_se"][index_t] = copy.deepcopy(mo_se)
                self.mf_statistics["mv"][index_t] = copy.deepcopy(mv)
                self.mf_statistics["mq"][index_t] = copy.deepcopy(mq)
                self.mf_statistics["mk"][index_t] = copy.deepcopy(mk)
                self.mf_statistics["m_tilde"][index_t] = copy.deepcopy(m_tilde)
                self.mf_statistics["m_tilde_proj"][index_t] = copy.deepcopy(m_tilde_proj)
                self.mf_statistics["m_pos"][index_t] = copy.deepcopy(m_pos)
            else:
                self.mf_statistics["mo"][index_t] = copy.deepcopy(mo._value)
                self.mf_statistics["mo_se"][index_t] = copy.deepcopy(mo_se._value)
                self.mf_statistics["mv"][index_t] = copy.deepcopy(mv._value)
                self.mf_statistics["mq"][index_t] = copy.deepcopy(mq._value)
                self.mf_statistics["mk"][index_t] = copy.deepcopy(mk._value)
                self.mf_statistics["m_tilde"][index_t] = copy.deepcopy(m_tilde._value)
                self.mf_statistics["m_tilde_proj"][index_t] = copy.deepcopy(m_tilde_proj._value)
                self.mf_statistics["m_pos"][index_t] = copy.deepcopy(m_pos._value)




    def init_mean_field(self, order, x0_idx):
        order = np.asarray(order)
        return (order == x0_idx).astype(float)

    def create_mf_window_from_correlation_idxs(self, x0_idx, ponder_pe=True):
        # TODO: have into account the PE in the initial condition?
        mv = self.init_mean_field(self.orders_B[self.feature_to_idx["v"]], x0_idx)[np.newaxis, :]
        mq = self.init_mean_field(self.orders_B[self.feature_to_idx["q"]], x0_idx)[np.newaxis, :]
        mk = self.init_mean_field(self.orders_B[self.feature_to_idx["k"]], x0_idx)[np.newaxis, :]

        return mv, mq, mk

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

        # Put in common queries and keys. mq[0] to unsqueeze the first dimension
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

    def all_sigma(self, m):
        indices = np.arange(2 ** m)
        return signed_binary_encoding(indices, m)


    def compute_tilde_m_no_trick(self, att_t_d):
        """
        att_t_d: (D, K)   = A_{t-1}
        where K is the extended M dimension
        returns: (C, D, K)
        """

        M = self.num_feat_patterns_se
        B_o = self.B_alpha[self.feature_to_idx["o"]]

        # ---- 1. shared field: B^o A ----
        # (D, K)
        h = np.einsum('kj,dj->dk', B_o, att_t_d)
        # equivalent to: h = att_t_d @ self.B_o.T

        # ---- 2. load projected sigma ----
        # (C, 2^M, K)
        sigma_proj_alpha = self.sigma_proj_alpha

        # ---- 3. dot product ----
        # (C, D, 2^M)
        dot = np.einsum('cnk,dk->cdn', sigma_proj_alpha, h)

        # ---- 4. nonlinearity ----
        # (C, D, 2^M)
        tanh_vals = np.tanh(self.beta_o * dot)

        # ---- 5. weighted uniform sum ----
        # (C, D, M)
        tilde_m = np.einsum('cdn,nk->cdk', tanh_vals, self.sigma) / (2 ** M)

        return tilde_m

    def compute_mfs(self, att_t_d, p_t_d):
        """
        Returns:
            dict with shapes:
                v, k: (D, M)
                q, o: (1, M)
            and o_se
        """

        N_P = p_t_d.shape[1]

        # ---- shared nonlinear core ----
        tilde_m = self.compute_tilde_m_no_trick(att_t_d)  # (C, D, K)

        tilde_m_proj = np.einsum('cdm,ckm->cdk', tilde_m, self.P_alpha)

        # ---- FIRST TERM (vectorized over features) ----
        # (C, D, K)
        term1_all = np.einsum('cij,cdj->cdi', self.B_alpha, tilde_m_proj)

        # ---- SECOND TERM ----
        # (C, D, K)
        term2_all = np.einsum('cki,di->cdk', self.W, p_t_d[:self.effective_context_size]) / N_P

        # ---- COMBINE ----
        m_all = self.epsilon_se * term1_all + self.epsilon_pe * term2_all  # (F, D, M)

        # ---- SPLIT FEATURES ----
        m_next = {}

        # v, k → full window
        for feat in ["v", "k"]:
            m_next[feat] = m_all[self.feature_to_idx[feat]]

        # q, o → only first timestep
        for feat in ["q", "o"]:
            m_next[feat] = m_all[self.feature_to_idx[feat], 0:1]  # 0:1 to maintain the desired dimensionality (1, K)

        # ---- extract o_se (no recomputation) ----
        o_se = term1_all[self.feature_to_idx["o"], 0:1]

        self.save_mf_stats(m_next["o"], o_se, m_next["v"][0], m_next["q"], m_next["k"][0], tilde_m[:,0],  term1_all[:,0], term2_all[:,0])

        return m_next["v"], m_next["q"], m_next["k"]



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

    def simulate(self, m0_idx, max_steps, compute_lyapunov=True):

        # Reconfigure PE to align with the initial pattern index
        # This ensures the cycle dynamics are properly aligned from the start
        self.reconfigure_pe_for_ini_m_idx(m0_idx)

        self.t = 0
        # Initialize attention with the info from the initial token
        mv, mq, mk = self.create_mf_window_from_correlation_idxs(m0_idx)
        # Create empty array for appending attention values
        att_t_d = np.array([]).reshape(0, self.num_feat_patterns)
        # Create attention
        att_t_d = self.attention(att_t_d, mv, mq, mk)
        # Initialize rotating PE
        p_t_d = self.PE.initialize_rotating_pe()

        # We simulate given a previously computed attention window and positional encoding

        self.effective_context_size = att_t_d.shape[0]
        self.t = self.effective_context_size

        # Flatten input for the jacobian
        att_t_1_d_flat = anp.reshape(att_t_d, att_t_d.shape[0] * self.num_feat_patterns)
        p_t_1_d_flat = anp.reshape(p_t_d, self.context_size * self.pe_bit_size)
        self.next_input = np.concatenate((att_t_1_d_flat, p_t_1_d_flat))

        if compute_lyapunov:
            # Define here the Jacobian handler over the function _step()
            self.Jacobian_Func = jacobian(self._step)
            dx = np.eye(self.num_feat_patterns * self.context_size + self.pe_bit_size * self.context_size)

        DEBUG_STEPS = 25000

        for t in range(0, max_steps):

            if t % DEBUG_STEPS == 0:
                print("Simulating step", t)

            self.t = t

            if not compute_lyapunov or (compute_lyapunov and (t < self.min_saved_step)):
                # If we don't want the gradients, just compute the output
                # _step() returns an output, but we also save it as self.next_input to save computation when
                # computing the gradients later
                self._step(self.next_input)

            if compute_lyapunov and (t >= self.min_saved_step):
                # Otherwise compute gradients and perturbations
                dx = self.lyapunov_step(self.next_input, dx)


            if self.t < self.context_size:
                # Increase previous context size + 1 until it reaches max
                self.effective_context_size = min(self.context_size, self.effective_context_size + 1)

            # else:
            #     self.effective_context_size = self.context_size

        if compute_lyapunov:
            self.lyapunov_end()
