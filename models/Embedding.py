import numpy as np
import copy
from utils import bitfield, bool2int


class Embedding:

    def __init__(self, se_bit_size, pe_bit_size):
        self.vocab_size = 2 ** se_bit_size
        self.se_bit_size = se_bit_size
        self.pe_bit_size = pe_bit_size
        self.initialized_pe = False

    def initialize(self):
        self.idx2word = np.zeros((self.vocab_size, self.se_bit_size + self.pe_bit_size))
        for i in range(self.vocab_size):
            self.idx2word[i, :self.se_bit_size] = (bitfield(i, self.se_bit_size) * 2) - 1

    def initialize_pos_encoder(self):
        self.pos2bit = np.zeros((2**self.pe_bit_size, self.pe_bit_size))
        for t in range(2**self.pe_bit_size):
            self.pos2bit[t, :] = self.encode_pos(t)

        self.initialized_pe = True


    def encode(self, idx):
        return copy.deepcopy(self.idx2word[idx])

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

    def encode_pos(self, pos):

        if self.initialized_pe:
            return self.pos2bit[pos, :]
        else:
            return bitfield(pos, self.pe_bit_size) * 2 - 1


    def decode(self, x):
        x = x[:self.se_bit_size]
        x = (x + 1) / 2
        return bool2int(x)
