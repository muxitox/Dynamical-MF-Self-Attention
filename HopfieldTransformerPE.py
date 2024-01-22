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

class Embedding:

    def __init__(self, se_bit_size, pe_bit_size):
        self.vocab_size = 2 ** se_bit_size
        self.se_bit_size = se_bit_size
        self.pe_bit_size = pe_bit_size

        self.idx2word = np.zeros((self.vocab_size, se_bit_size + pe_bit_size))

    def initialize(self):
        self.idx2word = np.zeros((self.vocab_size, self.se_bit_size + self.pe_bit_size))
        for i in range(self.vocab_size):
            self.idx2word[i, :self.se_bit_size] = (bitfield(i, self.se_bit_size) * 2) - 1

    def encode(self, idx):
        return self.idx2word[idx]


    def add_pe(self, x, pos):
        pe = bitfield(pos, self.pe_bit_size) * 2 - 1
        x[self.se_bit_size:] = pe
        return x

    def encode_w_pos(self, idx, pos):

        x = self.encode(idx)
        x = self.add_pe(x, pos)

        return x

    def decode(self, x):
        x = x[:self.se_bit_size]
        x = (x + 1)/2
        return bool2int(x)




class HopfieldTransformer:

    def __init__(self, beta_o, beta_att, num_feat_patterns, embedding_size, vocab, max_sim_steps=512):
        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = vocab.se_bit_size
        self.pe_bit_size = vocab.pe_bit_size

        self.W = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1

        reorder_weights = True
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

    def reset_data(self):
        self.x_list = np.zeros((max_sim_steps, embedding_size))

        self.mo_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mo_se_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mv_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mq_data = np.zeros((max_sim_steps, num_feat_patterns))
        self.mk_data = np.zeros((max_sim_steps, num_feat_patterns))

        self.att = np.zeros((max_sim_steps, num_feat_patterns))


    def qk_f(self, t, tau):

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
        # res2 = self.beta_att / np.sqrt(self.num_feat_patterns) * qk_accum
        # print(np.allclose(res, res2))

        return res


    def attention(self, t):

        key_prob = np.zeros(t+1)
        for tau in range(0, t+1):
            key_prob[tau] = self.qk_f(t, tau)
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
        return self.beta_att * self.embedding_size**2 / np.sqrt(self.num_feat_patterns) * mqk

    def attention_mf(self, t):

        key_prob = np.zeros(t+1)
        for tau in range(0, t+1):
            key_prob[tau] = self.exp_f_mf(t, tau)
        key_prob = softmax(key_prob)

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

        # Save for comparison with MF
        self.mo_data[0] = x0 @ self.Wo.T / self.embedding_size
        self.mo_se_data[0] = x0[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size

        selected_tokens = []

        for t in range(0, max_steps):
            att = self.attention(t)

            if t < max_steps-1:  # We'll compute att once more for computing statistics

                # We project all possible tokens in the vocabulary through Wo
                o = self.vocab.idx2word @ self.Wo.T

                # We multiply by the attention score
                prob_unnormalized = self.beta_o * o @ att

                prob_normalized = softmax(prob_unnormalized)

                print("Num tokens with max probability in time", t, ":", np.sum(np.isclose(prob_normalized,max(prob_normalized))), "/", self.vocab.vocab_size)


                # Convert the above result into a probability and get the idx of the most probable token
                sample = True
                if sample:
                    new_x_idx = np.random.choice(range(len(prob_normalized)), p=prob_normalized)
                else:
                    new_x_idx = np.argmax(prob_normalized)

                # Encode token and add it to the list
                new_x = self.vocab.encode(new_x_idx)
                self.x_list[t+1, :] = self.vocab.encode_w_pos(new_x_idx, t+1)

                # Save for comparison with MF
                self.mo_data[t+1] = new_x @ self.Wo.T / self.embedding_size
                self.mo_se_data[t+1] = new_x[:self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size

                selected_tokens.append(new_x_idx)

        return selected_tokens

    def compute_means_from_data(self, t):
        self.mo[t] = self.x_list[t] @ self.Wo.T / self.embedding_size
        self.mo_se[t] = self.x_list[t, :self.se_bit_size] @ self.Wo[:, :self.se_bit_size].T / self.embedding_size
        self.mv[t] = self.x_list[t] @ self.Wv.T / self.embedding_size
        self.mq[t] = self.x_list[t] @ self.Wq.T / self.embedding_size
        self.mk[t] = self.x_list[t] @ self.Wk.T / self.embedding_size

    def compute_mf(self, t, att):

        att_Wo_i = np.tanh(self.beta_o * np.einsum('b,bi -> i', att, self.Wo[:,:self.se_bit_size], optimize=True))
        att_Wo_i = np.concatenate((att_Wo_i, bitfield(t, self.pe_bit_size) * 2 - 1))
        self.mo[t] = np.einsum('bi,i ->b', self.Wo, att_Wo_i, optimize=True) / self.embedding_size
        self.mo_se[t] = np.einsum('bi,i ->b', self.Wo[:,:self.se_bit_size], att_Wo_i[:self.se_bit_size], optimize=True) / self.embedding_size

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
            att = self.attention_mf(t)

def plot_statistics_2_cols(stat1, stat2, stat_name, num_feat_patterns, num_plotting_steps):
    nrows = (num_feat_patterns + 1 ) // 2
    fig, ax = plt.subplots(nrows, 2,  figsize=(8, 2*nrows), constrained_layout=True)

    # if stat_name == "mo" or stat_name == "mo_se":
    #     num_plotting_steps_arange = np.arange(num_plotting_steps - 1)
    #     num_plotting_steps_arange += 1
    # else:
    #     num_plotting_steps_arange = np.arange(num_plotting_steps)

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

    # if stat_name == "mo" or stat_name == "mo_se":
    #     num_plotting_steps_arange = np.arange(num_plotting_steps - 1)
    #     num_plotting_steps_arange += 1
    # else:
    #     num_plotting_steps_arange = np.arange(num_plotting_steps)

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
    semantic_embedding_size = 14
    positional_embedding_size = 6
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    beta = 4
    beta_o = beta
    beta_att = beta

    num_feat_patterns = 6
    max_sim_steps = 20

    # Create seed for reproducibility
    # Nice seed for reorder of W (no more constrains), 14 se spins 6 pe spins 6 features: 10. Sample = True Interesting cycle.
    # Seed 13 (8, 16 + 6) does not coincide with std model
    seed = 10

    np.random.seed(seed)

    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns=num_feat_patterns, embedding_size=embedding_size,
                             vocab=vocab, max_sim_steps=max_sim_steps)

    # Select initial token
    random_idx = False
    if random_idx:
        x0_idx = 10  # You need to have an initial token to start decoding
        x0 = vocab.encode(x0_idx)
    else:
        x0 = HT.W[0]
        x0[semantic_embedding_size:] = -1  # Set embedding position for the first token to 0
        x0_idx = vocab.decode(x0)

        print(f"Initializing the model with the token with index {x0_idx}")

    print("List of tokens encoded in the features")
    print(HT.decoded_tokens)

    num_runs = 1

    mean_att = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mo_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mo_se_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mv_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mq_data = np.zeros((max_sim_steps, num_feat_patterns))
    mean_mk_data = np.zeros((max_sim_steps, num_feat_patterns))

    for i in range(0, num_runs):
        # Instantiate HT with the above created vocabulary

        # print("Simulating standard Transformer...")
        HT.reset_data()
        selected_tokens = HT.simulate(x0, max_steps=max_sim_steps, verbose=True)

        num_diff_tokens = len(np.unique(selected_tokens[10:]))
        # if num_diff_tokens > 2:
        #     print("Seed:", seed, "Num different tokens: ", num_diff_tokens)

        #  Collect data to compute the mean of the trajectory
        mean_att += HT.att
        mean_mo_data += HT.mo_data
        mean_mo_se_data += HT.mo_se_data
        mean_mv_data += HT.mv_data
        mean_mq_data += HT.mq_data
        mean_mk_data += HT.mk_data

    # Compute mean
    mean_att /= num_runs
    mean_mo_data /= num_runs
    mean_mo_se_data /= num_runs
    mean_mv_data /= num_runs
    mean_mq_data /= num_runs
    mean_mk_data /= num_runs

    print("Simulating MF Transformer...")
    HT.simulate_mf(x0, max_steps=max_sim_steps)
    print("Done.")

    # Plotting
    print("Plotting statistics...")
    num_plotting_steps = max_sim_steps
    plot_statistics(mean_att, HT.att_mf, "Att", num_feat_patterns, num_plotting_steps)

    plot_statistics(mean_mo_data, HT.mo, "mo", num_feat_patterns, num_plotting_steps)

    plot_statistics(mean_mo_se_data, HT.mo_se, "mo_se", num_feat_patterns, num_plotting_steps)

    plot_statistics(mean_mv_data, HT.mv, "mv", num_feat_patterns, num_plotting_steps)

    plot_statistics(mean_mq_data, HT.mq, "mq", num_feat_patterns, num_plotting_steps)

    plot_statistics(mean_mk_data, HT.mk, "mk", num_feat_patterns, num_plotting_steps)
    print("Done.")


