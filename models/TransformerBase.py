from abc import ABC, abstractmethod
import numpy as np

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


    @abstractmethod
    def simulate(self):
        pass