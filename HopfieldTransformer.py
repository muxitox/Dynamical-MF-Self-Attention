import numpy as np
import matplotlib.pyplot as plt
import os

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
        self.idx2word = []

    def initialize(self):
        self.idx2word = []

        for i in range(self.vocab_size):
            self.idx2word.append((bitfield(i, self.embedding_size) * 2) - 1)

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

        self.x_list = np.zeros((max_sim_steps, embedding_size))

    def exp_f(self, t, tau):
        accum = 0

        for a in range(0, self.num_feat_patterns):
            for i in range(0, self.embedding_size):
                for j in range(0, self.embedding_size):
                    accum += self.x_list[t,i] * self.Wq[i, a] * self.Wk[j, a] * self.x_list[tau,j]

        return np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * accum)

    def attention(self, b, t):
        att_b = 0
        for tau in range(0, t+1):
            for i in range(0, self.embedding_size):
                att_b += self.Wv[i,b] * self.x_list[tau][i] * self.exp_f(t, tau)

        Z = 0
        for tau in range(0, t+1):
            Z += self.exp_f(t, tau)

        att_b = att_b / Z

        return att_b

    def simulate(self, x0_idx, max_steps):

        x0 = self.vocab.encode(x0_idx)
        self.x_list[0,:] = x0

        for t in range(0, max_steps):
            for b in range(0, self.embedding_size):
                self.attention(b, t)





if __name__ == "__main__":


    # if not os.path.exists(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/"):
    #     os.makedirs(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/")

    beta_list = np.linspace(0,10, 500)

    m0 = 0.1

    last_m_odd_list = []
    last_m_even_list = []

    last_deriv_beta_odd = []
    last_deriv_beta_even = []

    beta_list_shallow = []

    embedding_size = 10
    vocab_size = 2**embedding_size
    vocab = Vocabulary(vocab_size, embedding_size)
    vocab.initialize()

    beta = 1
    beta_o = beta
    beta_att = beta
    x0_idx = 1
    num_feat_patterns = 10
    max_sim_steps = 512
    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns, embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps)
    HT.simulate(x0_idx, max_steps=max_sim_steps)
    # for beta in beta_list:
    #
    #
    #     NHT = NaiveHopfieldTransformer(beta, m0, Wodd, Weven)
    #
    #     max_steps = 512
    #     m_odd, m_even, deriv_beta_odd, deriv_beta_even = NHT.simulate(max_steps)
    #
    #     last_m_odd_list.append(m_odd[-1])
    #     last_m_even_list.append(m_even[-1])
    #
    #     last_deriv_beta_odd.append(deriv_beta_odd[-1])
    #     last_deriv_beta_even.append(deriv_beta_even[-1])
    #
    #     # plot_save_m_evolution(m_odd, m_even, beta, m0, Wodd, Weven)


