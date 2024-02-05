import numpy as np
from scipy.special import softmax


def bool2int(x):  # Transform bool array into positive integer
    """
    Transform bool array into positive integer. Code from
    https://github.com/MiguelAguilera/Adaptation-to-criticality-through-organizational-invariance/blob
    /dd46c3d272f05becaaf68bef92e724e5c3560150/Network/ising.py#L185
    :param x: :return:
    """
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2 ** i
    return int(y)


def bitfield(n, size):  # Transform positive integer into bit array
    x = [int(x) for x in bin(int(n))[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)


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

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, max_sim_steps=512, reorder_weights=False):
        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = vocab.se_bit_size
        self.pe_bit_size = vocab.pe_bit_size

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

        self.mo_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mo_se_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mv_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mq_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mk_data = np.zeros((max_sim_steps, num_feat_patterns))

        self.mo = np.zeros((max_sim_steps, num_feat_patterns))
        self.mo_se = np.zeros((max_sim_steps, num_feat_patterns))
        self.mv = np.zeros((max_sim_steps, num_feat_patterns))
        self.mq = np.zeros((max_sim_steps, num_feat_patterns))
        self.mk = np.zeros((max_sim_steps, num_feat_patterns))

        self.att = np.zeros((max_sim_steps, num_feat_patterns))
        self.att_mf = np.zeros((max_sim_steps, num_feat_patterns))

        self.x_list = np.zeros((max_sim_steps, embedding_size))

    def set_betas(self, beta_o, beta_att):
        self.beta_o = beta_o
        self.beta_att = beta_att

    def reset_data(self):
        self.x_list = np.zeros((self.max_sim_steps, self.embedding_size))

        self.mo_data = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mo_se_data = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mv_data = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mq_data = np.zeros((self.max_sim_steps, self.num_feat_patterns))
        self.mk_data = np.zeros((self.max_sim_steps, self.num_feat_patterns))

        self.att = np.zeros((self.max_sim_steps, self.num_feat_patterns))

    def qk_f(self, t, tau):

        q = self.x_list[t] @ self.Wq.T  # Query representation
        k = self.Wk @ self.x_list[tau]  # Key representation

        # Save the statistics for comparison with the MF approximation
        if t == tau:
            self.mq_data[t] = q / self.embedding_size
            self.mk_data[t] = k / self.embedding_size

        qk = q @ k

        res = np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * qk)

        # # Loopy implementation for testing
        # qk_accum = 0
        # for a in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         for j in range(0, self.embedding_size):
        #             qk_accum += self.x_list[t,i] * self.Wq[a, i] * self.Wk[a, j] * self.x_list[tau,j]
        #
        # res2 = self.beta_att / np.sqrt(self.num_feat_patterns) * qk_accum
        # print(np.allclose(res, res2))

        return res

    def attention(self, t):

        key_prob = np.zeros(t + 1)
        for tau in range(0, t + 1):
            key_prob[tau] = self.qk_f(t, tau)
        key_prob /= np.sum(key_prob)

        v = self.x_list[0:t + 1] @ self.Wv.T  # Value representation
        att_t = key_prob @ v

        # Save for stats comparison
        self.mv_data[t] = v[-1] / self.embedding_size

        # # Loopy implementation for testing
        # att_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         for tau in range(0, t + 1):
        #             att_t[b] += self.Wv[b, i] * self.x_list[tau,i] * key_prob[tau]

        self.att[t] = att_t

        return att_t

    def qk_f_mf(self, t, tau):

        mqk = self.mq[t] @ self.mk[tau]
        return self.beta_att * self.embedding_size ** 2 / np.sqrt(self.num_feat_patterns) * mqk

    def attention_mf(self, t):

        key_prob = np.zeros(t + 1)
        for tau in range(0, t + 1):
            key_prob[tau] = self.qk_f_mf(t, tau)
        key_prob = softmax(key_prob)

        att_t = self.embedding_size * (self.mv[:t + 1].T @ key_prob)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        self.att_mf[t] = att_t

        return att_t

    def attention_mf_optimized(self, t):

        mqk = np.einsum('b,tb -> t', self.mq[t], self.mk[:t + 1, :], optimize=True)

        key_prob = self.beta_att * self.embedding_size ** 2 / np.sqrt(self.num_feat_patterns) * mqk
        key_prob = softmax(key_prob)

        att_t = self.embedding_size * (self.mv[:t + 1].T @ key_prob)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        self.att_mf[t] = att_t

        return att_t

    def simulate(self, x0, max_steps, verbose=False):

        self.x_list[0, :] = x0

        # Save for comparison with MF
        self.mo_data[0] = x0 @ self.Wo.T / self.embedding_size
        self.mo_se_data[0] = x0[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size

        selected_tokens = []

        for t in range(0, max_steps):
            att = self.attention(t)

            if t < max_steps - 1:  # We'll compute att once more for computing statistics

                # We project all possible tokens in the vocabulary through Wo
                o = self.vocab.idx2word @ self.Wo.T

                # We multiply by the attention score
                prob_unnormalized = self.beta_o * o @ att

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
                new_x = self.vocab.encode(new_x_idx)
                self.x_list[t + 1, :] = self.vocab.encode_w_pos(new_x_idx, t + 1)

                # Save for comparison with MF
                self.mo_data[t + 1] = new_x @ self.Wo.T / self.embedding_size
                self.mo_se_data[t + 1] = new_x[:self.se_bit_size] @ self.Wo[:,
                                                                    :self.se_bit_size].T / self.embedding_size

                selected_tokens.append(new_x_idx)

        return selected_tokens

    def compute_means_from_data(self, t):
        self.mo[t] = self.x_list[t] @ self.Wo.T / self.embedding_size
        self.mo_se[t] = self.x_list[t, :self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size
        self.mv[t] = self.x_list[t] @ self.Wv.T / self.embedding_size
        self.mq[t] = self.x_list[t] @ self.Wq.T / self.embedding_size
        self.mk[t] = self.x_list[t] @ self.Wk.T / self.embedding_size

    def compute_mf(self, t, att):

        # Compute the mean of every (semantic) spin i at time t
        att_Wo_i = np.tanh(self.beta_o * np.einsum('b,bi -> i', att, self.Wo[:, :self.se_bit_size], optimize=True))
        # Concatenate semantic information with positional encoding
        att_Wo_i = np.concatenate((att_Wo_i, bitfield(t, self.pe_bit_size) * 2 - 1))
        # Compute mean fields
        self.mo[t] = np.einsum('bi,i ->b', self.Wo, att_Wo_i, optimize=True) / self.embedding_size
        # Compute only semantic information
        self.mo_se[t] = np.einsum('bi,i ->b', self.Wo[:, :self.se_bit_size], att_Wo_i[:self.se_bit_size],
                                  optimize=True) / self.embedding_size

        # # Loopy implementation for testing
        # mo_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         mo_t[b] += self.Wo[b, i] * np.tanh(self.beta_o * self.Wo[:, i] @ att)
        # mo_t /= self.embedding_size
        # print(np.allclose(self.mo[t], mo_t))

        self.mv[t] = np.einsum('bi,i ->b', self.Wv, att_Wo_i, optimize=True) / self.embedding_size
        self.mq[t] = np.einsum('bi,i ->b', self.Wq, att_Wo_i, optimize=True) / self.embedding_size
        self.mk[t] = np.einsum('bi,i ->b', self.Wk, att_Wo_i, optimize=True) / self.embedding_size

    def simulate_mf(self, x0, max_steps):
        self.x_list[0, :] = x0

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(t=0)
        att = self.attention_mf(t=0)

        for t in range(1, max_steps):
            self.compute_mf(t, att)
            att = self.attention_mf_optimized(t)
