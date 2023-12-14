import numpy as np
import matplotlib.pyplot as plt
import os
# Create seed for reproducibility
np.random.seed(1)

imgs_root = "imgs/hopfield_transformer"


def bool2int(x):  # Transform bool array into positive integer
    """
    Transform bool array into positive integer. Code from
    https://github.com/MiguelAguilera/Adaptation-to-criticality-through-organizational-invariance/blob
    /dd46c3d272f05becaaf68bef92e724e5c3560150/Network/ising.py#L185
    :param x: :return:
    """
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2**i
    return y


def bitfield(n,size):			# Transform positive integer into bit array
    x = [int(x) for x in bin(int(n))[2:]]
    x = [0]*(size-len(x)) + x
    return np.array(x)


class Vocabulary:

    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.idx2word = np.zeros((vocab_size, embedding_size))

    def initialize(self):
        self.idx2word = np.zeros((vocab_size, embedding_size))

        for i in range(self.vocab_size):
            self.idx2word[i] = (bitfield(i, self.embedding_size) * 2) - 1

    def encode(self, idx):
        return self.idx2word[idx]


class HopfieldTransformer:

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, max_sim_steps=512):
        self.beta_o = beta_o
        self.beta_att = beta_att
        self.Wo = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
        self.Wv = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
        self.Wq = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
        self.Wk = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
        self.num_feat_patterns = num_feat_patterns
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.max_sim_steps = max_sim_steps

        self.mo = np.zeros((max_sim_steps, num_feat_patterns))
        self.mv = np.zeros((max_sim_steps, num_feat_patterns))
        self.mq = np.zeros((max_sim_steps, num_feat_patterns))
        self.mk = np.zeros((max_sim_steps, num_feat_patterns))

        self.x_list = np.zeros((max_sim_steps, embedding_size))

    def exp_f(self, t, tau):

        q = self.x_list[t] @ self.Wq.T  # Query representation
        k = self.Wk @ self.x_list[tau]  # Key representation
        qk = q @ k

        return np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * qk)

        # # Loopy implementation for testing
        # qk_accum = 0
        # for a in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         for j in range(0, self.embedding_size):
        #             qk_accum += self.x_list[t,i] * self.Wq[a, i] * self.Wk[a, j] * self.x_list[tau,j]
        #
        # return np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * qk_accum)

    def attention(self, t):

        key_prob = np.zeros(t+1)
        for tau in range(0, t+1):
            key_prob[tau] = self.exp_f(t, tau)
        key_prob /= np.sum(key_prob)

        v = self.x_list[0:t+1] @ self.Wv.T  # Value representation
        att_t = key_prob @ v

        # # Loopy implementation for testing
        # att_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         for tau in range(0, t + 1):
        #             att_t[b] += self.Wv[b, i] * self.x_list[tau,i] * key_prob[tau]

        return att_t

    def exp_f_mf(self, t, tau):

        # q = self.x_list[t] @ self.Wq.T  # Query representation
        # k = self.Wk @ self.x_list[tau]  # Key representation
        # qk = q @ k
        #
        # return np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * qk)

        # Loopy implementation for testing
        accum = 0
        for a in range(0, self.num_feat_patterns):
            accum += self.mq[t, a] * self.mq[tau, a]

        return np.exp(self.beta_att * self.embedding_size**2 / np.sqrt(self.num_feat_patterns) * accum)

    def attention_mf(self, t):

        key_prob = np.zeros(t+1)
        for tau in range(0, t+1):
            key_prob[tau] = self.exp_f_mf(t, tau)
        key_prob /= np.sum(key_prob)

        att_t = self.embedding_size * (self.mv[:t+1].T @ key_prob)

        # # Loopy implementation for testing
        #
        # att_t_loopy = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for tau in range(0, t+1):
        #         att_t_loopy[b] += self.embedding_size * self.mv[tau, b] * key_prob[tau]

        return att_t

    def simulate(self, x0_idx, max_steps):

        x0 = self.vocab.encode(x0_idx)
        self.x_list[0,:] = x0

        for t in range(0, max_steps):
            att = self.attention(t)

            # We project all possible tokens in the vocabulary through Wo
            o = self.vocab.idx2word @ self.Wo.T

            # We multiply by the attention score
            prob_unnormalized = o @ att

            # Convert the above result into a probability and get the idx of the most probable token
            new_x_idx = np.argmax(prob_unnormalized)

            # Select it and add it to the list
            self.x_list[t+1, :] = self.vocab.encode(new_x_idx)

            print(f'In position {t} we have selected token {new_x_idx}')

    def compute_means_from_data(self, t):
        self.mv[t] = self.x_list[t] @ self.Wv.T / self.embedding_size
        self.mq[t] = self.x_list[t] @ self.Wq.T / self.embedding_size
        self.mk[t] = self.x_list[t] @ self.Wk.T / self.embedding_size

    def compute_mf(self, t, att):

        att_i = np.tanh( self.beta_o * np.einsum('i,e -> ei', att, np.ones(self.embedding_size)) @ self.Wo)
        self.mo[t] = np.einsum('bi,ii ->b', self.Wk, att_i)/self.embedding_size

        # # Loopy implementation for testing
        # mo_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         mo_t[b] += self.Wk[b, i] * np.tanh(self.beta_o * self.Wo[:, i] @ att)
        # mo_t /= self.embedding_size
        # print(np.allclose(mo2, mo_t))

        self.mv[t] = np.einsum('bi,ii ->b', self.Wv, att_i)/self.embedding_size
        self.mq[t] = np.einsum('bi,ii ->b', self.Wq, att_i)/self.embedding_size
        self.mq[t] = np.einsum('bi,ii ->b', self.Wk, att_i)/self.embedding_size

    def simulate_mf(self, x0_idx, max_steps):

        x0 = self.vocab.encode(x0_idx)
        self.x_list[0, :] = x0

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(t=0)
        att = self.attention_mf(t=0)

        for t in range(1, max_steps):
            self.compute_mf(t, att)
            att = self.attention_mf(t)


if __name__ == "__main__":

    # if not os.path.exists(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/"):
    #     os.makedirs(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/")

    # Instantiate vocabulary
    embedding_size = 8
    vocab_size = 2**embedding_size
    vocab = Vocabulary(vocab_size, embedding_size)
    vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    beta = 1
    beta_o = beta
    beta_att = beta
    x0_idx = 1  # You need to have an initial token to start decoding
    num_feat_patterns = 10
    max_sim_steps = 512
    # Instantiate HT with the above created vocabulary
    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns, embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps)
    # HT.simulate(x0_idx, max_steps=max_sim_steps)
    HT.simulate_mf(x0_idx, max_steps=max_sim_steps)



