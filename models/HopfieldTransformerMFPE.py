import copy
import numpy as np
from scipy.special import softmax
from models.TransformerBase import TransformerBase
class HopfieldTransformer:

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, context_size, max_sim_steps=512,
                 min_saved_step=0,
                 normalize_weights_str_att="N**2", normalize_weights_str_o="N", reorder_weights=False, pe_mode=0,
                 weights_from_segments=False, scaling_o=1, scaling_att=1):
        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = vocab.se_bit_size
        self.pe_bit_size = vocab.pe_bit_size
        self.embedding_size = self.se_bit_size + self.pe_bit_size
        self.context_size = context_size
        self.context_index = 0

        self.scaling_o = scaling_o
        self.scaling_att = scaling_att

        N = embedding_size
        M = num_feat_patterns
        self.normalize_weights_str_att = normalize_weights_str_att
        self.normalize_weights_str_o = normalize_weights_str_o
        self.total_normalization_o = self.define_total_normalization_o()
        self.total_normalization_att = self.define_total_normalization_att()
        # Dynamically compute the normalize_weights_str string
        # Dynamically compute the normalize_weights_str string
        try:
            exec_str = f"self.normalizing_constant_att = {self.normalize_weights_str_att}"
            exec_str2 = f"self.normalizing_constant_o = {self.normalize_weights_str_o}"
            exec(exec_str)
            exec(exec_str2)
        except:
            print("Either of the exec_str for the normalizing_constants is not well defined")
            raise

        self.W = np.zeros((num_feat_patterns, embedding_size))
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
            self.Wo = np.zeros((num_feat_patterns, embedding_size))
            self.Wv = np.zeros((num_feat_patterns, embedding_size))
            self.Wq = np.zeros((num_feat_patterns, embedding_size))
            self.Wk = np.zeros((num_feat_patterns, embedding_size))

            self.Wo_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wv_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wq_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wk_SE = np.random.randint(2, size=(num_feat_patterns, self.se_bit_size)) * 2 - 1

            self.Wo[:, :self.se_bit_size] = self.Wo_SE
            self.Wv[:, :self.se_bit_size] = self.Wv_SE
            self.Wq[:, :self.se_bit_size] = self.Wq_SE
            self.Wk[:, :self.se_bit_size] = self.Wk_SE

            if pe_mode == 1 or (pe_mode == 0 and weights_from_segments):
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

        num_segments_corrs = 3
        if weights_from_segments:  #  create uniform +1 -1 segments and combine them

            segment_size = self.se_bit_size / num_segments_corrs

            pe_num_segments = int(self.pe_bit_size / segment_size) + 1
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
                            segment_end_pe = int(embedding_size - pe_segment_id * segment_size + 1)
                            segment_begin_pe = max(self.se_bit_size, int(self.pe_bit_size - (pe_segment_id + 1)
                                                                         * segment_size))

                            segment_begin = int((pe_segment_id + segments_diff) * segment_size)

                            curr_W[i, segment_begin_pe:segment_end_pe] = curr_W[
                                i, segment_begin]  # Initialize PE to its corresponding segment


            # Create correlations from matrices for comparison
            if num_feat_patterns < 1 or num_feat_patterns > 3:
                raise "The number of patterns is neither 1, 2 or 3"

            self.pair_corr_o_o = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_v = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_k = np.zeros((num_feat_patterns, num_feat_patterns))
            self.pair_corr_o_q = np.zeros((num_feat_patterns, num_feat_patterns))

            for b in range(0, num_feat_patterns):
                for a in range(0, num_feat_patterns):
                    for i in range(0, self.se_bit_size):
                        self.pair_corr_o_o[a, b] += self.Wo[a, i] * self.Wo[b, i]
                        self.pair_corr_o_v[a, b] += self.Wo[a, i] * self.Wv[b, i]
                        self.pair_corr_o_k[a, b] += self.Wo[a, i] * self.Wk[b, i]
                        self.pair_corr_o_q[a, b] += self.Wo[a, i] * self.Wq[b, i]

            self.pair_corr_o_o /= self.se_bit_size
            self.pair_corr_o_v /= self.se_bit_size
            self.pair_corr_o_k /= self.se_bit_size
            self.pair_corr_o_q /= self.se_bit_size

            if num_feat_patterns == 3:
                self.quad_corr_o_o = np.zeros(num_feat_patterns)
                self.quad_corr_o_v = np.zeros(num_feat_patterns)
                self.quad_corr_o_k = np.zeros(num_feat_patterns)
                self.quad_corr_o_q = np.zeros(num_feat_patterns)

                for b in range(0, num_feat_patterns):
                    for i in range(0, self.se_bit_size):
                        Wo_corr = self.Wo[0, i] * self.Wo[1, i] * self.Wo[2, i]
                        self.quad_corr_o_o[b] += Wo_corr * self.Wo[b, i]
                        self.quad_corr_o_v[b] += Wo_corr * self.Wv[b, i]
                        self.quad_corr_o_q[b] += Wo_corr * self.Wq[b, i]
                        self.quad_corr_o_k[b] += Wo_corr * self.Wk[b, i]

                self.quad_corr_o_o /= self.se_bit_size
                self.quad_corr_o_v /= self.se_bit_size
                self.quad_corr_o_q /= self.se_bit_size
                self.quad_corr_o_k /= self.se_bit_size

        self.decoded_tokens = np.zeros(len(self.W))

        for a in range(0, len(self.W)):
            self.decoded_tokens[a] = vocab.decode(self.W[a])

        # In 2D same behavior as 2feat
        self.num_feat_patterns = num_feat_patterns
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.max_sim_steps = max_sim_steps
        self.min_saved_step = min_saved_step


        # Create variables for saving the statistics of the standard model
        self.std_statistics = {}
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        for name_i in self.statistics_names:
            self.std_statistics[name_i] = np.zeros((self.num_saved_statistics, num_feat_patterns))

        # List to save selected tokens in the standard model execution
        self.x_list = np.zeros((self.num_saved_statistics, embedding_size))

        # Create variables for saving the statistics of the mean-field model
        self.mf_statistics = {}
        for name_i in self.statistics_names:
            self.mf_statistics[name_i] = np.zeros((self.num_saved_statistics, num_feat_patterns))

        self.x_window = np.zeros((self.context_size, embedding_size))


    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def reset_data(self):
        self.x_list = np.zeros((self.max_sim_steps, self.embedding_size))

        for name_i in self.statistics_names:
            self.std_statistics[name_i] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
            self.mf_statistics[name_i] = np.zeros((self.max_sim_steps, self.num_feat_patterns))

    def reset_data_keep_context(self):
        x_list_copy = copy.deepcopy(self.x_list)

        mf_statistics_copy = {}
        std_statistics_copy = {}
        for name_i in self.statistics_names:
            mf_statistics_copy[name_i] = copy.deepcopy(self.mf_statistics[name_i])
            std_statistics_copy[name_i] = copy.deepcopy(self.std_statistics[name_i])

        self.reset_data()


        self.x_list[:self.context_size, :] = x_list_copy[-self.context_size:, :]
        for name_i in self.statistics_names:
            self.mf_statistics[name_i][:self.context_size, :] = mf_statistics_copy[name_i][-self.context_size:, :]
            self.std_statistics[name_i][:self.context_size, :] = std_statistics_copy[name_i][-self.context_size:, :]


    def define_total_normalization_o(self):
        total_normalization = self.embedding_size / self.normalizing_constant_o * self.scaling_o

        return total_normalization

    def define_total_normalization_att(self):
        total_normalization = self.embedding_size ** 2 / self.normalizing_constant_att * self.scaling_att

        return total_normalization

    def attention_mf_unoptimized(self, t):

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

    def attention_mf(self, t):

        effective_context_size = min(self.context_size, t + 1)

        mqk = np.einsum('b,tb -> t', self.mq_window, self.mk_window[:effective_context_size],
                        optimize=True)

        key_prob_unnorm = self.beta_att * self.total_normalization_att * mqk

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

    def compute_means_from_data(self, t):
        self.mf_statistics["mo"][t] = self.x_list[t] @ self.Wo.T / self.embedding_size
        self.mf_statistics["mo_se"][t] = self.x_list[t, :self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.se_bit_size
        self.mf_statistics["mv"][t] = self.x_list[t] @ self.Wv.T / self.embedding_size
        self.mf_statistics["mq"][t] = self.x_list[t] @ self.Wq.T / self.embedding_size
        self.mf_statistics["mk"][t] = self.x_list[t] @ self.Wk.T / self.embedding_size

    def compute_mf(self, t, att):

        # Compute the mean of every (semantic) spin i at time t
        att_Wo_i = np.tanh(self.beta_o * (1 / self.normalizing_constant) *
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
            att = self.attention_mf(t)

    def simulate_mf(self, x0, max_steps):
        self.x_list[0, :] = x0

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(t=0)
        att = self.attention_mf(t=0)

        for t in range(1, max_steps):
            self.compute_mf(t, att)
            att = self.attention_mf(t)
