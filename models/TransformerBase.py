from abc import ABC, abstractmethod
import numpy as np
import copy

class TransformerBase(ABC):

    def __init__(self, beta_o, beta_att, num_feat_patterns, positional_embedding_bitsize, vocab, context_size, N,
                 max_sim_steps=512, min_saved_step=0, normalize_weights_str_att="N**2", normalize_weights_str_o="N",
                 reorder_weights=False, pe_mode=0, semantic_embedding_bitsize=0, scaling_o=1, scaling_att=1):

        self.beta_o = beta_o
        self.beta_att = beta_att
        self.se_bit_size = semantic_embedding_bitsize
        self.pe_bit_size = positional_embedding_bitsize
        self.vocab = vocab

        self.pe_mode = pe_mode

        self.PE = self.PositionalEncoding(positional_embedding_bitsize, vocab, K=20, type="tanh")

        self.embedding_size = semantic_embedding_bitsize + positional_embedding_bitsize

        self.context_size = context_size
        self.num_feat_patterns = num_feat_patterns
        self.max_sim_steps = max_sim_steps
        self.min_saved_step = min_saved_step
        self.num_saved_steps = max_sim_steps - min_saved_step

        self.scaling_o = scaling_o
        self.scaling_att = scaling_att

        self.reorder_weights = reorder_weights

        M = num_feat_patterns
        self.normalize_weights_str_att = normalize_weights_str_att
        self.normalize_weights_str_o = normalize_weights_str_o


        # Dynamically compute the normalize_weights_str strings
        try:
            exec_str = f"self.normalizing_constant_att = {self.normalize_weights_str_att}"
            exec_str2 = f"self.normalizing_constant_o = {self.normalize_weights_str_o}"
            exec(exec_str)
            exec(exec_str2)
        except:
            raise Exception("Either of the exec_str for the normalizing_constants is not well defined")

        # Initialize matrices

        self.Wo = np.zeros((self.num_feat_patterns, self.embedding_size))
        self.Wv = np.zeros((self.num_feat_patterns, self.embedding_size))
        self.Wq = np.zeros((self.num_feat_patterns, self.embedding_size))
        self.Wk = np.zeros((self.num_feat_patterns, self.embedding_size))

        self.pair_corr_o_o = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
        self.pair_corr_o_v = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
        self.pair_corr_o_k = np.zeros((self.num_feat_patterns, self.num_feat_patterns))
        self.pair_corr_o_q = np.zeros((self.num_feat_patterns, self.num_feat_patterns))

        if self.num_feat_patterns == 3:
            self.quad_corr_o_o = np.zeros(self.num_feat_patterns)
            self.quad_corr_o_v = np.zeros(self.num_feat_patterns)
            self.quad_corr_o_k = np.zeros(self.num_feat_patterns)
            self.quad_corr_o_q = np.zeros(self.num_feat_patterns)

    def create_W_matrices_finite_model(self, weights_from_segments, num_segments_corrs):

        self.W = np.zeros((self.num_feat_patterns, self.embedding_size))
        self.W_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
        self.W[:, :self.se_bit_size] = self.W_SE
        self.W[:, -self.pe_bit_size:] = self.W_SE[:, -self.pe_bit_size:]

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
            # self.Wk = self.Wq

        else:
            self.Wo_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wv_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wq_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1
            self.Wk_SE = np.random.randint(2, size=(self.num_feat_patterns, self.se_bit_size)) * 2 - 1

            self.Wo[:, :self.se_bit_size] = self.Wo_SE
            self.Wv[:, :self.se_bit_size] = self.Wv_SE
            self.Wq[:, :self.se_bit_size] = self.Wq_SE
            self.Wk[:, :self.se_bit_size] = self.Wk_SE

            if self.pe_mode == 1 or (self.pe_mode == 0 and weights_from_segments):
                # If pe_mode==0 and correlations_from_weighst=3. We use this (its like setting them random below but found seeds are more interesting)
                self.Wo[:, -self.pe_bit_size:] = self.Wo_SE[:, -self.pe_bit_size:]
                self.Wv[:, -self.pe_bit_size:] = self.Wv_SE[:, -self.pe_bit_size:]
                self.Wq[:, -self.pe_bit_size:] = self.Wq_SE[:, -self.pe_bit_size:]
                self.Wk[:, -self.pe_bit_size:] = self.Wk_SE[:, -self.pe_bit_size:]
            elif self.pe_mode == 0:
                self.Wo[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(self.num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wv[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(self.num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wq[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(self.num_feat_patterns, self.pe_bit_size)) * 2 - 1
                self.Wk[:, -self.pe_bit_size:] = np.random.randint(2,
                                                                   size=(self.num_feat_patterns, self.pe_bit_size)) * 2 - 1

            self.W = self.Wo

        matrix_list = [self.Wo, self.Wv, self.Wq, self.Wk]

        if weights_from_segments:  # create uniform +1 -1 segments and combine them

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
                            segment_begin_pe = max(self.se_bit_size, int(self.pe_bit_size - (pe_segment_id + 1)
                                                                         * segment_size))

                            segment_begin = int((pe_segment_id + segments_diff) * segment_size)

                            curr_W[i, segment_begin_pe:segment_end_pe] = curr_W[
                                i, segment_begin]  # Initialize PE to its corresponding segment

    def define_pair_correlations_from_weights(self):

        for b in range(0, self.num_feat_patterns):
            for a in range(0, self.num_feat_patterns):
                for i in range(0, self.se_bit_size):
                    self.pair_corr_o_o[a, b] += self.Wo[a, i] * self.Wo[b, i]
                    self.pair_corr_o_v[a, b] += self.Wo[a, i] * self.Wv[b, i]
                    self.pair_corr_o_k[a, b] += self.Wo[a, i] * self.Wk[b, i]
                    self.pair_corr_o_q[a, b] += self.Wo[a, i] * self.Wq[b, i]

        self.pair_corr_o_o /= self.se_bit_size
        self.pair_corr_o_v /= self.se_bit_size
        self.pair_corr_o_k /= self.se_bit_size
        self.pair_corr_o_q /= self.se_bit_size


    def define_quad_correlations_from_weights(self):

        for b in range(0, self.num_feat_patterns):
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

        self.even_corr_o_o = copy.deepcopy(self.pair_corr_o_o)
        self.even_corr_o_v = copy.deepcopy(self.pair_corr_o_v)
        self.even_corr_o_k = copy.deepcopy(self.pair_corr_o_k)
        self.even_corr_o_q = copy.deepcopy(self.pair_corr_o_q)

        self.even_corr_o_o = np.vstack((self.pair_corr_o_o, self.quad_corr_o_o))
        self.even_corr_o_v = np.vstack((self.pair_corr_o_v, self.quad_corr_o_v))
        self.even_corr_o_k = np.vstack((self.pair_corr_o_k, self.quad_corr_o_k))
        self.even_corr_o_q = np.vstack((self.pair_corr_o_q, self.quad_corr_o_q))

    class PositionalEncoding:
        def __init__(self, pe_bit_size, vocab, K=1, type="base"):
            # type can be "base" or "tanh"
            self.pe_bit_size = pe_bit_size
            self.state = np.ones(pe_bit_size, dtype=np.longdouble) * -1
            self.K = K
            self.vocab = vocab
            self.type = type
            self.dp_dp = np.zeros((pe_bit_size, pe_bit_size))

        def initialize_state(self, t):
            self.state = self.vocab.encode_pos(t)

        def getter(self):
            return self.state

        def next_step(self, compute_der=True):
            new_state = np.zeros(self.pe_bit_size, dtype=np.longdouble)

            new_state[-1] = - self.state[-1]
            if self.type == "tanh":
                new_state[-1] *= self.K

            for i in range(self.pe_bit_size-2, -1, -1):
                new_state[i] = new_state[i+1] * self.state[i]

            # If in tanh mode, we can save computation by computing the derivative here
            if self.type == "tanh" and compute_der:
                dp_dp_0 = np.einsum("j,k->jk", new_state, 1/self.state)

                der_tanh = 1 - np.tanh(new_state)**2
                dp_dp = np.einsum("jk,j->jk", dp_dp_0, der_tanh)
                self.dp_dp = np.tril(dp_dp)

            # Save state
            self.state = copy.deepcopy(new_state)



            if self.type == "tanh":
                self.state = np.tanh(self.state)

    @abstractmethod
    def attention(self, t):
        pass

    @abstractmethod
    def simulate(self, x0, max_steps):
        pass