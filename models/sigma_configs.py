import numpy as np
from utils import signed_binary_encoding

def set_orders_manually_4_duplicity_loop_1(self, ini_m_idx=None):
    # Designing to try to get a loop if ini_m_idx=0 (not used but hardcoded for the moment)

    # Set SE order of patterns
    # order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    order_in_B = np.array([0, 1, 2, 3])
    order_out_B = np.array([1, 2, 1, 0]) # 0 → 1, 1 → 2, 2 → 1, 3 → 0 Both should be equivalent?
    # order_out_B = np.array([1, 2, 3, 0]) # 0 → 1, 1 → 2, 2 → 3, 3 → 0

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2, 1])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    # order_in_pe = np.array([0, 1, 2, 3])
    # order_out_pe = np.copy(order_out)
    # order_out_pe = np.array([2, 0, 3, 1])

    # order_in_pe = np.array([1, 0, 2, 3])
    # order_out_pe = np.array([2, 3, 0, 1])

    order_in_pe = np.array([2, 0, 1, 3])
    order_out_pe = np.array([3, 1, 0, 2])

    # order_in_pe = np.array([3, 0, 1, 2])
    # order_out_pe = np.array([2, 3, 0, 1])



    # order_in_pe = np.roll(order_in_pe, 1)
    # order_out_pe = np.roll(order_out_pe, 1)

    print("order in", order_in_pe)
    print("order out", order_out_pe)

    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W

def set_orders_manually_4_duplicity_loop_2(self, ini_m_idx=None):
    # Designing to try to get a loop if ini_m_idx=0 (not used but hardcoded for the moment)
    # This design alternates patterns but does not get them to 1.

    # Set SE order of patterns
    # order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    order_in_B = np.array([0, 1, 2, 3])
    order_out_B = np.array([1, 2, 3, 0])  # shift +1 # 0 → 1, 1 → 2, 2 → 3, 3 → 0

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2, 1])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    order_in_pe = np.array([2, 0, 1, 3])
    order_out_pe = np.array([2, 3, 0, 1])

    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W

def set_orders_manually_4_duplicity_loop_3(self, ini_m_idx=None):
    # Try and get only one active pattern in mean fields at every moment.

    # Set SE order of patterns
    # order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    order_in_B = np.array([0, 1, 2, 3])
    order_out_B = np.array([1, 2, 3, 0])  # shift +1 # 0 → 1, 1 → 2, 2 → 3, 3 → 0

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2, 1])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    order_in_pe = np.array([2, 3, 1, 0])
    order_out_pe = np.array([3, 1, 0, 2])

    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W


def set_orders_manually_4_duplicity_loop_4(self, ini_m_idx=None):
    # Designing to try to get a loop if ini_m_idx=0 (not used but hardcoded for the moment)

    # Set SE order of patterns
    # order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    order_in_B = np.array([0, 1, 2, 3])
    # order_out_B = np.array([1, 2, 1, 0]) # 0 → 1, 1 → 2, 2 → 1, 3 → 0 Both should be equivalent?
    order_out_B = np.array([1, 2, 3, 0]) # 0 → 1, 1 → 2, 2 → 3, 3 → 0

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2, 1])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    order_in_pe = np.array([3, 0, 1, 2])
    order_out_pe = np.array([2, 3, 0, 1])


    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W

def set_orders_manually_4_duplicity_loop_no_pe(self, ini_m_idx=None):
    # This one is only like a 3 feat loop, but they are not alterned. You have to set epsilon_pe to 0

    # Set SE order of patterns
    # order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    order_in_B = np.array([0, 1, 2, 1])
    order_out_B = np.array([2, 0, 1, 3])  # shift +1
    # order_out_B = np.array([3, 0, 1, 2])  # shift +1, same m^o behavior, different A behavior

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2, 1])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    order_in_pe = np.array([2, 0, 1, 3])
    order_out_pe = np.array([2, 3, 0, 1])

    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W


def set_orders_manually_4_no_duplicity(self, ini_m_idx=None):
    # Set SE order of patterns
    order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    # order_in = np.array([0, 1, 2, 3])
    # order_out = np.array([1, 2, 3, 0]) # shift -1
    order_out_B = np.array([3, 0, 1, 2])  # shift +1

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2, 3])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    order_in_pe = np.copy(order_in_B)
    # order_out_pe = np.copy(order_out)
    order_out_pe = np.array([3, 0, 1, 2])

    order_in_pe = np.roll(order_in_pe, 1)
    order_out_pe = np.roll(order_out_pe, 1)

    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    # w_rng = np.random.default_rng(self.seed_W)
    # W = w_rng.integers(0, 2, (len(self.features_names), self.num_feat_patterns, self.pe_bit_size)) * 2 - 1

    return orders_B, orders_P, W

def set_orders_manually_4_cycle_0_1_2_1(self, ini_m_idx=0):
    """
    GENERATED BY COPILOT AND WRONG

    Configure a cycle: 0 → 1 → 2 → 1 (repeats)
    
    With 3 base semantic patterns duplicated to 4 total attention patterns,
    where attention positions 1 and 3 encode the same semantic pattern.
    
    Key design:
    - order_in_B: full permutation [0,1,2,3] - ALL attention positions contribute to tilde_m
    - order_out_B: cycle routing [1,2,1,0] - routes dynamics through the cycle
      * Position 0 (pattern 0) → goes to position 1 (pattern 1)
      * Position 1 (pattern 1) → goes to position 2 (pattern 2)
      * Position 2 (pattern 2) → goes to position 1 (pattern 1, cycle back)
      * Position 3 (also pattern 1) → goes to position 0 (pattern 0)
    
    - order_P: [0,1,2,1] - semantic duplication: positions 1 and 3 both decode pattern 1
      This means 2 of the 4 attention values represent the same semantic meaning
    
    - order_in_pe: [0,1,2,3] - PE encodes cycle indices (input, aligned with ini_m_idx)
    - order_out_pe: [1,2,1,0] - PE output follows dynamics (shifted to maximize next position)
    
    Args:
        ini_m_idx: initial pattern index (0, 1, 2, or 3). PE encoding is aligned 
                   so that the next position in the cycle gets maximum contribution.
    """
    # Set SE order of patterns - MUST be a full permutation for all attention to contribute!
    order_in_B = np.array([0, 1, 2, 3])  # Full permutation: all 4 positions map to themselves
    
    # Output order: route through the cycle
    # 0 → 1, 1 → 2, 2 → 1, 3 → 0
    order_out_B = np.array([1, 2, 1, 0])

    orders_B = np.stack([
        order_in_B,  # q - queries: identity permutation
        order_in_B,  # k - keys: identity permutation
        order_out_B,  # v - values: follow output cycle routing
        order_in_B,  # o - output: identity permutation
    ], axis=0)  # shape: (C, M)

    # Pattern projection: semantic duplication (3 base patterns represented by 4 positions)
    # Positions 1 and 3 both encode the same semantic pattern
    order_P = np.array([0, 1, 2, 1])  # 3 base patterns: 0, 1, 2, and 1 duplicated at pos 3
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    # PE (Positional Encoding) mapping aligned with initial condition
    # The cycle trajectory: 0 → 1 → 2 → 1 → 0 → ...
    # Align so that if we start at ini_m_idx, next position is maximally encoded
    order_in_pe = np.array([0, 1, 2, 3])   # PE input: encode all 4 positions
    base_cycle_out = np.array([1, 2, 1, 0])  # Base cycle routing
    
    # Shift order_out_pe so that the next position after ini_m_idx gets max encoding
    # This ensures m_tilde and attention peak at the correct next position
    order_out_pe = np.roll(base_cycle_out, -ini_m_idx)
    order_in_pe = np.roll(order_in_pe, -ini_m_idx)

    print("=== Cycle 0 → 1 → 2 → 1 Configuration ===")
    print(f"Initial pattern index (ini_m_idx): {ini_m_idx}")
    print("Base cycle: position 0→1→2→1→0→1→...")
    print(f"order_in_B: {order_in_B} (full permutation - all attention values contribute)")
    print(f"order_out_B: {order_out_B} (cycle routing)")
    print(f"order_P: {order_P} (semantic: patterns 0,1,2,1 from 4 positions)")
    print(f"order_in_pe (shifted):  {order_in_pe}")
    print(f"order_out_pe (shifted): {order_out_pe}")
    print(f"Next position after {ini_m_idx}: {base_cycle_out[ini_m_idx]}")

    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W



def set_orders_manually_6_duplicity_loop(self, ini_m_idx=None):
    # Set SE order of patterns
    # order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    order_in_B = np.array([0, 1, 2, 3, 1, 3])
    order_out_B = np.array([3, 0, 1, 2, 4, 5])  # shift +1
    # order_out_B = np.array([2, 0, 1, 2])  # shift +1, same m^o behavior, different A behavior

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2, 3, 1, 3])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    order_in_pe = np.copy(order_in_B)
    # order_out_pe = np.copy(order_out)
    order_out_pe = np.array([3, 0, 1, 2, 4, 5])

    # order_in_pe = np.roll(order_in_pe, 1)
    # order_out_pe = np.roll(order_out_pe, 1)

    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)



    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W

def set_orders_manually_3_no_duplicity(self):
    # Set SE order of patterns
    order_in_B = np.arange(self.num_feat_patterns)
    # order_out_B = np.roll(order_in_B, 1)

    # order_in = np.array([0, 1, 2, 3])
    # order_out = np.array([1, 2, 3, 0]) # shift -1
    order_out_B = np.array([2, 0, 1])  # shift +1

    orders_B = np.stack([
        order_in_B,  # q
        order_in_B,  # k
        order_out_B,  # v
        order_in_B,  # o
    ], axis=0)  # shape: (C, M)

    order_P = np.array([0, 1, 2])
    orders_P = np.stack([
        order_P,  # q
        order_P,  # k
        order_P,  # v
        order_P,  # o
    ], axis=0)  # shape: (C, M)

    order_in_pe = np.copy(order_in_B)
    # order_out_pe = np.copy(order_out)
    order_out_pe = order_out_B
    # Set up PE
    pe_in = signed_binary_encoding(order_in_pe, self.pe_bit_size)
    pe_out = signed_binary_encoding(order_out_pe, self.pe_bit_size)

    # Create W matrix to group all features together for the vectorized computation of the mean-field values
    W = np.zeros((len(self.features_names), self.num_feat_patterns, self.pe_bit_size))
    W[self.feature_to_idx["q"]] = pe_in
    W[self.feature_to_idx["k"]] = pe_in
    W[self.feature_to_idx["v"]] = pe_out
    W[self.feature_to_idx["o"]] = pe_in

    return orders_B, orders_P, W