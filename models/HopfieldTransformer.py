import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
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
    return int(y)


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

    def decode(self, x):
        x = (x + 1)/2
        return bool2int(x)


class HopfieldTransformer:

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, max_sim_steps=512):
        self.beta_o = beta_o
        self.beta_att = beta_att

        self.W = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1

        self.decoded_tokens = np.zeros(len(self.W))

        for a in range(0, len(self.W)):
            self.decoded_tokens[a] = vocab.decode(self.W[a])


        self.Wo = np.copy(self.W)
        np.random.shuffle(self.Wo)
        # self.Wv = np.copy(self.W)
        # np.random.shuffle(self.Wv)
        self.Wv = np.roll(self.Wo, 1, 1)
        self.Wq = np.copy(self.W)
        np.random.shuffle(self.Wq)
        # self.Wk = np.copy(self.W)
        # np.random.shuffle(self.Wk)
        self.Wk = self.Wq

        # In 2D same behavior as 2feat

        self.num_feat_patterns = num_feat_patterns
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.max_sim_steps = max_sim_steps

        self.mo_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mv_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mq_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mk_data = np.zeros((max_sim_steps, num_feat_patterns))

        self.mo = np.zeros((max_sim_steps, num_feat_patterns))
        self.mv = np.zeros((max_sim_steps, num_feat_patterns))
        self.mq = np.zeros((max_sim_steps, num_feat_patterns))
        self.mk = np.zeros((max_sim_steps, num_feat_patterns))

        self.att = np.zeros((max_sim_steps, num_feat_patterns))
        self.att_mf = np.zeros((max_sim_steps, num_feat_patterns))

        self.x_list = np.zeros((max_sim_steps, embedding_size))

    def exp_f(self, t, tau):

        q = self.x_list[t] @ self.Wq.T  # Query representation
        k = self.Wk @ self.x_list[tau]  # Key representation

        # Save the statistics for comparison with the MF approximation
        if t==tau:
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
        # res2 = np.exp(self.beta_att / np.sqrt(self.num_feat_patterns) * qk_accum)
        # print(np.allclose(res, res2))

        return res


    def attention(self, t):

        key_prob = np.zeros(t+1)
        for tau in range(0, t+1):
            key_prob[tau] = self.exp_f(t, tau)
        key_prob /= np.sum(key_prob)

        v = self.x_list[0:t+1] @ self.Wv.T  # Value representation
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

    def exp_f_mf(self, t, tau):

        mqk = self.mq[t] @ self.mk[tau]
        return np.exp(self.beta_att * self.embedding_size**2 / np.sqrt(self.num_feat_patterns) * mqk)

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

        self.att_mf[t] = att_t

        return att_t

    def simulate(self, x0, max_steps, verbose=False):
        self.x_list[0,:] = x0

        selected_tokens = []

        for t in range(0, max_steps):
            att = self.attention(t)

            if t < max_steps-1:  # We'll compute att once more for computing statistics

                # We project all possible tokens in the vocabulary through Wo
                o = self.vocab.idx2word @ self.Wo.T

                # We multiply by the attention score
                prob_unnormalized = self.beta_o * o @ att

                print("Num tokens with max probability", np.sum(prob_unnormalized==max(prob_unnormalized)), "/", self.vocab.vocab_size)

                prob_normalized = softmax(prob_unnormalized)

                # Convert the above result into a probability and get the idx of the most probable token
                new_x_idx = np.random.choice(range(len(prob_normalized)), p=prob_normalized)

                # Encode token and add it to the list
                new_x = self.vocab.encode(new_x_idx)
                self.x_list[t+1, :] = self.vocab.encode(new_x_idx)
                # Save for comparison with MF
                self.mo_data[t+1] = new_x @ self.Wo.T / self.embedding_size

                selected_tokens.append(new_x_idx)

                if verbose:
                    print(f'In position {t+1} we have selected token {new_x_idx}')

        return selected_tokens

    def compute_means_from_data(self, t):
        self.mv[t] = self.x_list[t] @ self.Wv.T / self.embedding_size
        self.mq[t] = self.x_list[t] @ self.Wq.T / self.embedding_size
        self.mk[t] = self.x_list[t] @ self.Wk.T / self.embedding_size

    def compute_mf(self, t, att):

        att_i = np.tanh(self.beta_o * np.einsum('b,bi -> i', att, self.Wo, optimize=True))
        self.mo[t] = np.einsum('bi,i ->b', self.Wo, att_i, optimize=True) / self.embedding_size

        # # Loopy implementation for testing
        # mo_t = np.zeros(self.num_feat_patterns)
        # for b in range(0, self.num_feat_patterns):
        #     for i in range(0, self.embedding_size):
        #         mo_t[b] += self.Wo[b, i] * np.tanh(self.beta_o * self.Wo[:, i] @ att)
        # mo_t /= self.embedding_size
        # print(np.allclose(self.mo[t], mo_t))

        self.mv[t] = np.einsum('bi,i ->b', self.Wv, att_i, optimize=True) / self.embedding_size
        self.mq[t] = np.einsum('bi,i ->b', self.Wq, att_i, optimize=True) / self.embedding_size
        self.mk[t] = np.einsum('bi,i ->b', self.Wk, att_i, optimize=True) / self.embedding_size

    def simulate_mf(self, x0, max_steps):
        self.x_list[0, :] = x0

        # Initialize attention with the info from the initial token
        self.compute_means_from_data(t=0)
        att = self.attention_mf(t=0)

        for t in range(1, max_steps):
            self.compute_mf(t, att)
            att = self.attention_mf(t)

def plot_statistics_2_cols(stat1, stat2, stat_name, num_feat_patterns, num_plotting_steps):
    nrows = (num_feat_patterns + 1 ) // 2
    fig, ax = plt.subplots(nrows, 2,  figsize=(8, 2*nrows), constrained_layout=True)

    if stat_name == "mo":
        num_plotting_steps_arange = np.arange(num_plotting_steps - 1)
        num_plotting_steps_arange += 1
    else:
        num_plotting_steps_arange = np.arange(num_plotting_steps)

    for i in range(0, min(10, num_feat_patterns)):

        row = i // 2
        if num_feat_patterns <= 2:
            local_ax = ax[i % 2]
        else:
            local_ax = ax[row, i % 2]
        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, i], label="std")
        local_ax.plot(num_plotting_steps_arange, stat2[:num_plotting_steps, i], '--', label="mf")
        if i > 3:
            local_ax.set_xlabel("t")
        local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    fig.suptitle(f"Evolution of {stat_name}")
    plt.show()

def plot_statistics_1d(stat1, stat2, stat_name, num_plotting_steps):

    if stat_name == "mo":
        num_plotting_steps_arange = np.arange(num_plotting_steps - 1)
        num_plotting_steps_arange += 1
    else:
        num_plotting_steps_arange = np.arange(num_plotting_steps)

    plt.figure()
    plt.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, 0], label="std")
    plt.plot(num_plotting_steps_arange, stat2[:num_plotting_steps, 0], '--', label="mf")
    plt.xlabel("t")
    plt.legend(loc="upper right")

    plt.title(f"Evolution of {stat_name}")
    plt.show()

def plot_statistics(stat1, stat2, stat_name, num_feat_patterns, num_plotting_steps):

    if num_feat_patterns == 1:
        plot_statistics_1d(stat1, stat2, stat_name, num_plotting_steps)
    else:
        plot_statistics_2_cols(stat1, stat2, stat_name, num_feat_patterns, num_plotting_steps)

if __name__ == "__main__":

    # if not os.path.exists(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/"):
    #     os.makedirs(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/")

    # Instantiate vocabulary
    embedding_size = 16
    vocab_size = 2**embedding_size
    vocab = Vocabulary(vocab_size, embedding_size)
    vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    beta = 4
    beta_o = beta
    beta_att = beta

    num_feat_patterns = 4
    max_sim_steps = 20

    # Create seed for reproducibility
    # Embedding size 4:
    # Seed for 3 length cycle with 3 patterns: 9
    # Seed for 4 length cycle with 4 patterns: 19, 33, 41, 53 Only the 4% were 4 lenght cycles. Only 3% were 3 length. (Out of 100)
    # Embedding size 16:
    # Seed for 4 length cycle with 4 patterns: 106. A different initial state than W[0] changes the dynamic

    seed = 1

    np.random.seed(seed)

    # Instantiate HT with the above created vocabulary
    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns, embedding_size=embedding_size, vocab=vocab, max_sim_steps=max_sim_steps)

    # Select initial token
    random_idx = False
    if random_idx:
        x0_idx = 10  # You need to have an initial token to start decoding
        x0 = vocab.encode(x0_idx)
    else:
        x0 = HT.W[0]
        x0_idx = vocab.decode(x0)
        print(f"Initializing the model with the token with index {x0_idx}")

    print("List of tokens encoded in the features")
    print(HT.decoded_tokens)

    print("Simulating standard Transformer...")
    selected_tokens = HT.simulate(x0, max_steps=max_sim_steps, verbose=True)
    print("Simulating MF Transformer...")
    HT.simulate_mf(x0, max_steps=max_sim_steps)
    print("Done.")

    num_diff_tokens = len(np.unique(selected_tokens[10:]))
    if num_diff_tokens > 2:
        print("Seed:", seed, "Num different tokens: ", num_diff_tokens)

    # Plotting
    print("Plotting statistics...")
    num_plotting_steps = max_sim_steps
    plot_statistics(HT.att, HT.att_mf, "Att", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mo_data[1:], HT.mo[1:], "mo", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mv_data, HT.mv, "mv", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mq_data, HT.mq, "mq", num_feat_patterns, num_plotting_steps)

    plot_statistics(HT.mk_data, HT.mk, "mk", num_feat_patterns, num_plotting_steps)
    print("Done.")


