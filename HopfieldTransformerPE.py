import numpy as np
from scipy.special import softmax
from utils import bitfield, bool2int


class Embedding:

    def __init__(self, se_bit_size, pe_bit_size):
        self.vocab_size = 2 ** se_bit_size
        self.se_bit_size = se_bit_size
        self.pe_bit_size = pe_bit_size

    def initialize(self):
        self.idx2word = np.zeros((self.vocab_size, self.se_bit_size + self.pe_bit_size))
        for i in range(self.vocab_size):
            self.idx2word[i, :self.se_bit_size] = (bitfield(i, self.se_bit_size) * 2) - 1

    def encode(self, idx):
        return self.idx2word[idx]

    def encode_force(self, idx, pos):
        # Make position with modular arithmetics to avoid a possible error
        pos = pos % self.pe_bit_size ** 2

        se = bitfield(idx, self.se_bit_size) * 2 - 1
        pe = bitfield(pos, self.pe_bit_size) * 2 - 1

        return np.concatenate((se, pe))

    def add_pe(self, x, pos):
        # Make position with modular arithmetics to avoid a possible error
        pos = pos % self.pe_bit_size ** 2
        pe = bitfield(pos, self.pe_bit_size) * 2 - 1
        x[self.se_bit_size:] = pe
        return x

    def encode_w_pos(self, idx, pos):
        x = self.encode(idx)
        x = self.add_pe(x, pos)

        return x

    def decode(self, x):
        x = x[:self.se_bit_size]
        x = (x + 1) / 2
        return bool2int(x)


class HopfieldTransformer:

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, context_size, max_sim_steps=512,
                 normalize_weights_str="1", reorder_weights=False):
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

        self.W = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1


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
            self.Wo = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
            self.Wv = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
            self.Wq = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
            self.Wk = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1

            self.W = self.Wo

        self.decoded_tokens = np.zeros(len(self.W))

        for a in range(0, len(self.W)):
            self.decoded_tokens[a] = vocab.decode(self.W[a])

        # In 2D same behavior as 2feat

        self.num_feat_patterns = num_feat_patterns
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.max_sim_steps = max_sim_steps

        # Create variables for saving the statistics of the standard model
        self.std_statistics = {}
        self.statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
        for name_i in self.statistics_names:
            self.std_statistics[name_i] = np.zeros((max_sim_steps, num_feat_patterns))

        # List to save selected tokens in the standard model execution
        self.x_list = np.zeros((max_sim_steps, embedding_size))

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
            self.std_statistics[name_i] = np.zeros((self.max_sim_steps, self.num_feat_patterns))
            self.mf_statistics[name_i] = np.zeros((self.max_sim_steps, self.num_feat_patterns))

    def qk_f(self, t, tau):

        q = self.x_list[t] @ self.Wq.T  # Query representation
        k = self.Wk @ self.x_list[tau]  # Key representation

        # Save the statistics for comparison with the MF approximation
        if t == tau:
            self.std_statistics["mq"][t] = q / self.embedding_size
            self.std_statistics["mk"][t] = k / self.embedding_size

        qk = q @ k

        # res = np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * qk)
        res = np.exp(self.beta_att * (1 / self.normalizing_constant) * qk)


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

        idx_ctx_start = max(0, t - self.context_size + 1)
        effective_context_size = min(self.context_size, t + 1)

        key_prob = np.zeros(effective_context_size)
        for tau in range(0, effective_context_size):
            key_prob[tau] = self.qk_f(t, idx_ctx_start + tau)
        key_prob /= np.sum(key_prob)

        v = self.x_list[idx_ctx_start:t + 1] @ self.Wv.T  # Value representation
        att_t = key_prob @ v

        # Save for stats comparison
        self.std_statistics["mv"][t] = v[-1] / self.embedding_size

        # # Loopy implementation for testing
        # att_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         for tau in range(0, t + 1):
        #             att_t[b] += self.Wv[b, i] * self.x_list[tau,i] * key_prob[tau]

        self.std_statistics["att"][t] = att_t

        return att_t

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

    def simulate(self, x0, max_steps, verbose=False):

        self.x_list[0, :] = x0

        # Save for comparison with MF
        self.std_statistics["mo"][0] = x0 @ self.Wo.T / self.embedding_size
        self.std_statistics["mo_se"][0] = x0[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size

        selected_tokens = []

        for t in range(0, max_steps):
            att = self.attention(t)

            if t < max_steps - 1:  # We'll compute att once more for computing statistics

                # We project all possible tokens in the vocabulary through Wo
                o = self.vocab.idx2word @ self.Wo.T

                # We multiply by the attention score
                prob_unnormalized = self.beta_o * (1 / self.normalizing_constant) * o @ att

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
                self.std_statistics["mo"][t + 1] = self.x_list[t + 1, :] @ self.Wo.T / self.embedding_size
                self.std_statistics["mo_se"][t + 1] = self.x_list[t + 1, :][:self.se_bit_size] @ self.Wo[:,
                                                                    :self.se_bit_size].T / self.embedding_size

                selected_tokens.append(new_x_idx)

        return selected_tokens

    def compute_means_from_data(self, t):
        self.mf_statistics["mo"][t] = self.x_list[t] @ self.Wo.T / self.embedding_size
        self.mf_statistics["mo_se"][t] = self.x_list[t, :self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size
        self.mf_statistics["mv"][t] = self.x_list[t] @ self.Wv.T / self.embedding_size
        self.mf_statistics["mq"][t] = self.x_list[t] @ self.Wq.T / self.embedding_size
        self.mf_statistics["mk"][t] = self.x_list[t] @ self.Wk.T / self.embedding_size

    def compute_mf(self, t, att):

        # Compute the mean of every (semantic) spin i at time t
        att_Wo_i = np.tanh(self.beta_o * (1 / self.normalizing_constant) * np.einsum('b,bi -> i', att, self.Wo[:, :self.se_bit_size], optimize=True))
        # Concatenate semantic information with positional encoding
        att_Wo_i = np.concatenate((att_Wo_i, bitfield(t % self.context_size, self.pe_bit_size) * 2 - 1))
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

        self.mf_statistics["mv"][t] = np.einsum('bi,i ->b', self.Wv, att_Wo_i, optimize=True) / self.embedding_size
        self.mf_statistics["mq"][t] = np.einsum('bi,i ->b', self.Wq, att_Wo_i, optimize=True) / self.embedding_size
        self.mf_statistics["mk"][t] = np.einsum('bi,i ->b', self.Wk, att_Wo_i, optimize=True) / self.embedding_size

    def simulate_mf(self, x0, max_steps):
        self.x_list[0, :] = x0

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(t=0)
        att = self.attention_mf(t=0)

        for t in range(1, max_steps):
            self.compute_mf(t, att)
            att = self.attention_mf_optimized(t)
