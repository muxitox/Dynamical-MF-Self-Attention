import copy
import numpy as np
from scipy.special import softmax
from utils import bitfield, bool2int


class HopfieldTransformerPEML:

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, context_size, max_sim_steps=512,
                 normalize_weights_str="1", reorder_weights=False, pe_mode=0):
        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = vocab.se_bit_size
        self.pe_bit_size = vocab.pe_bit_size

        self.context_size = context_size

        N = embedding_size
        M = num_feat_patterns

        # Dynamically compute the normalize_weights_str string
        try:
            exec_str = f"self.normalizing_constant = {normalize_weights_str}"
            exec(exec_str)
        except:
            print("The exec_str string is not well defined")
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

            if pe_mode == 0:
                self.Wo[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wv[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wq[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wk[:, -self.pe_bit_size:] = np.random.randint(2, size=(num_feat_patterns, self.pe_bit_size)) * 2 - 1

            else:
                self.Wo[:, -self.pe_bit_size:] = self.Wo_SE[:, -self.pe_bit_size:]
                self.Wv[:, -self.pe_bit_size:] = self.Wv_SE[:, -self.pe_bit_size:]
                self.Wq[:, -self.pe_bit_size:] = self.Wq_SE[:, -self.pe_bit_size:]
                self.Wk[:, -self.pe_bit_size:] = self.Wk_SE[:, -self.pe_bit_size:]

            self.W = self.Wo

        self.decoded_tokens = np.zeros(len(self.W))

        for a in range(0, len(self.W)):
            self.decoded_tokens[a] = vocab.decode(self.W[a])

        # In 2D same behavior as 2feat

        self.num_feat_patterns = num_feat_patterns
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.max_sim_steps = max_sim_steps

        # List to save selected tokens in the standard model execution
        self.x_list = np.zeros((max_sim_steps, embedding_size))

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


    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def reset_data(self):
        self.x_list = np.zeros((self.max_sim_steps, self.embedding_size))

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
        x_list_copy = copy.deepcopy(self.x_list)

        mf_statistics_copy = {}
        for name_i in self.statistics_names:
            mf_statistics_copy[name_i] = copy.deepcopy(self.mf_statistics[name_i])

        self.reset_data()

        self.x_list[:self.context_size, :] = x_list_copy[-self.context_size:, :]
        for name_i in self.statistics_names:
            self.mf_statistics[name_i][self.context_size] = mf_statistics_copy[name_i][1]


    def qk_f_mf(self, t, d):
        mqk = self.mf_statistics["mq"][t] @ self.mf_statistics["mk"][t, d, :]
        return self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk

    def attention_mf(self, t):

        effective_context_size = min(self.context_size, t + 1)

        key_prob = np.zeros(effective_context_size)
        for d in range(0, effective_context_size):
            key_prob[d] = self.qk_f_mf(t, d)
        key_prob = softmax(key_prob)

        att_t = self.embedding_size * (self.mf_statistics["mv"][t, :effective_context_size,:].T @ key_prob)

        self.mf_statistics["att"][t] = att_t

        return att_t

    def attention_mf_optimized(self, t):

        effective_context_size = min(self.context_size, t + 1)

        mqk = np.einsum('b,tb -> t', self.mf_statistics["mq"][t], self.mf_statistics["mk"][t, :effective_context_size, :], optimize=True)

        # key_prob = self.beta_att * self.embedding_size ** 2 / np.sqrt(self.num_feat_patterns) * mqk
        key_prob = self.beta_att * (1 / self.normalizing_constant) * self.embedding_size ** 2 * mqk
        # key_prob = self.beta_att * self.embedding_size ** 2 * mqk

        key_prob = softmax(key_prob)

        att_t = self.embedding_size * (self.mf_statistics["mv"][t,:effective_context_size].T @ key_prob)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        self.mf_statistics["att"][t] = att_t

        return att_t

    def compute_means_from_data(self, t):
        # self.mf_statistics["mo"][t] = self.x_list[t] @ self.Wo.T / self.embedding_size
        # self.mf_statistics["mo_se"][t] = self.x_list[t, :self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.se_bit_size
        # self.mf_statistics["mv"][t] = self.x_list[t] @ self.Wv.T / self.embedding_size
        # self.mf_statistics["mq"][t] = self.x_list[t] @ self.Wq.T / self.embedding_size
        # self.mf_statistics["mk"][t] = self.x_list[t] @ self.Wk.T / self.embedding_size

        self.mf_statistics["mo"][0] = self.x_list[t] @ self.Wo.T / self.embedding_size
        self.mf_statistics["mo_se"][0] = self.x_list[t, :self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.se_bit_size
        self.mf_statistics["mv"][0, 0, :] = self.x_list[t] @ self.Wv.T / self.embedding_size
        self.mf_statistics["mq"][0] = self.x_list[t] @ self.Wq.T / self.embedding_size
        self.mf_statistics["mk"][0, 0, :] = self.x_list[t] @ self.Wk.T / self.embedding_size

    def shift_d_window(self, t):
        # Roll the window of d copies by 1 position
        self.mf_statistics["mv"][t] = np.roll(self.mf_statistics["mv"][t-1], 1, axis=0)
        self.mf_statistics["mk"][t] = np.roll(self.mf_statistics["mk"][t-1], 1, axis=0)
        self.mf_statistics["pe"][t] = np.roll(self.mf_statistics["pe"][t-1], 1, axis=0)


    def compute_mf(self, t, att):

        # Compute the mean of every (semantic) spin i at time t
        att_Wo_i = np.tanh(self.beta_o * (1 / self.normalizing_constant) * np.einsum('b,bi -> i', att, self.Wo[:, :self.se_bit_size], optimize=True))
        # Concatenate semantic information with positional encoding
        att_Wo_i = np.concatenate((att_Wo_i, self.mf_statistics["pe"][t,0,:]))

        # Compute mean fields
        self.mf_statistics["mo"][t] = np.einsum('bi,i ->b', self.Wo, att_Wo_i, optimize=True) / self.embedding_size
        # Compute only semantic information
        self.mf_statistics["mo_se"][t] = np.einsum('bi,i ->b', self.Wo[:, :self.se_bit_size], att_Wo_i[:self.se_bit_size],
                                     optimize=True) / self.se_bit_size

        # # Loopy implementation for testing
        # mo_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         mo_t[b] += self.Wo[b, i] * np.tanh(self.beta_o * (1 / self.normalizing_constant) *  self.Wo[:, i] @ att)
        # mo_t /= self.embedding_size
        # print(np.allclose(self.mo[t], mo_t))

        self.mf_statistics["mv"][t, 0, :] = np.einsum('bi,i ->b', self.Wv, att_Wo_i, optimize=True) / self.embedding_size
        self.mf_statistics["mq"][t] = np.einsum('bi,i ->b', self.Wq, att_Wo_i, optimize=True) / self.embedding_size
        self.mf_statistics["mk"][t, 0, :] = np.einsum('bi,i ->b', self.Wk, att_Wo_i, optimize=True) / self.embedding_size


    def simulate_mf_from_context(self, max_steps):
        # In order for this method to work properly, a simulate_mf() method has had to be run previously at least for
        # self.context_size steps

        # Initialize attention to the last computed attention
        att = self.mf_statistics["att"][self.context_size - 1, :]

        # We initialize the model at the end of the previous execution
        ini_t = self.context_size
        for t in range(ini_t, max_steps):
            self.shift_d_window(t)

            self.compute_mf(t, att)

            att = self.attention_mf_optimized(t)

    def simulate_mf(self, x0, max_steps):
        self.x_list[0, :] = x0

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(t=0)
        # att = self.attention_mf_optimized(t=0)
        att = self.attention_mf_optimized(t=0)

        for t in range(1, max_steps):
            self.shift_d_window(t)

            self.compute_mf(t, att)

            # att = self.attention_mf_optimized(t)
            att = self.attention_mf_optimized(t)

