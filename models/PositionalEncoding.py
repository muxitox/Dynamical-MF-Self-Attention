import numpy as np
import autograd.numpy as anp
import copy

class PositionalEncoding:
    def __init__(self, pe_bit_size, vocab, context_size, K=1, type="base"):
        # type can be "base" or "tanh"
        self.pe_bit_size = pe_bit_size
        # self.state = np.ones(pe_bit_size, dtype=np.longdouble) * -1
        self.state = np.ones(pe_bit_size) * -1
        self.state_window = anp.zeros((context_size, self.pe_bit_size))
        self.state_window[0] = copy.deepcopy(self.state)
        self.context_size = context_size

        self.K = K
        self.vocab = vocab
        self.type = type
        self.dp_dp = np.zeros((pe_bit_size, pe_bit_size))

    def initialize_rotating_pe(self):

        pe_window = np.zeros((self.context_size, self.pe_bit_size))

        for i in range(self.context_size):
            pe_window[i] = self.vocab.encode_pos(i)

        pe_window = pe_window[::-1]
        pe_window = np.roll(pe_window, 1, axis=0)

        return pe_window

    def initialize_state(self, t):
        self.state = self.vocab.encode_pos(t)

    def getter(self):
        return self.state

    def next_step_autograd(self, p_t_1_d):

        # prev_state = p_t_1_d[0]
        #
        #
        # new_state = - prev_state[-1]
        # if self.type == "tanh":
        #     new_state *= self.K
        #
        #
        # new_state = anp.array([new_state])
        #
        # for i in range(self.pe_bit_size - 2, -1, -1):
        #     new_bit = new_state[i] * self.state[i]
        #     new_state = np.hstack((new_state, new_bit))
        #
        # # new_state2 =  (- anp.cumprod((prev_state[::-1])) )[::-1]
        #
        #
        # if self.type == "tanh":
        #     # new_state2 = anp.tanh(new_state2 * self.K)
        #     new_state = anp.tanh(new_state)

        p_t_d = anp.roll(p_t_1_d, 1, axis=0)
        # p_t_d[0, :] = new_state

        return p_t_d

    def next_step(self, compute_der=True):
        # new_state = np.zeros(self.pe_bit_size, dtype=np.longdouble)
        new_state = np.zeros(self.pe_bit_size)

        new_state[-1] = - self.state[-1]
        if self.type == "tanh":
            new_state[-1] *= self.K

        for i in range(self.pe_bit_size - 2, -1, -1):
            new_state[i] = new_state[i + 1] * self.state[i]

        # If in tanh mode, we can save computation by computing the derivative here
        if self.type == "tanh":
            if compute_der:
                dp_dp_0 = np.einsum("j,k->jk", new_state, 1 / self.state)

            new_state = np.tanh(new_state)

            if compute_der:
                der_tanh = 1 - new_state ** 2
                dp_dp = np.einsum("jk,j->jk", dp_dp_0, der_tanh)
                # Get the upper anti-diagonal to 0, so future time-steps do not influence current in the derivative
                self.dp_dp = np.flipud(np.triu(np.flipud(dp_dp)))

        self.state_window = np.roll(self.state_window, 1, axis=0)
        self.state_window[0] = copy.deepcopy(new_state)

        # Save state
        self.state = copy.deepcopy(new_state)