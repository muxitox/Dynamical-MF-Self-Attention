import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

seed = 8
semantic_embedding_bitsize = 100
positional_embedding_bitsize = 10

num_feat_patterns = M = 50
embedding_size = N = semantic_embedding_bitsize + positional_embedding_bitsize

np.random.seed(seed)
Wo = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
Wv = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
Wq = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1
Wk = np.random.randint(2, size=(num_feat_patterns, embedding_size)) * 2 - 1

pair_corr_o_o = np.zeros((num_feat_patterns, num_feat_patterns))
pair_corr_o_v = np.zeros((num_feat_patterns, num_feat_patterns))
pair_corr_o_k = np.zeros((num_feat_patterns, num_feat_patterns))
pair_corr_o_q = np.zeros((num_feat_patterns, num_feat_patterns))

for b in range(0, num_feat_patterns):
    for a in range(0, num_feat_patterns):
        for i in range(0, semantic_embedding_bitsize):
            pair_corr_o_o[a, b] += Wo[a, i] * Wo[b, i]
            pair_corr_o_v[a, b] += Wo[a, i] * Wv[b, i]
            pair_corr_o_k[a, b] += Wo[a, i] * Wk[b, i]
            pair_corr_o_q[a, b] += Wo[a, i] * Wq[b, i]

pair_corr_o_o /= semantic_embedding_bitsize
pair_corr_o_v /= semantic_embedding_bitsize
pair_corr_o_q /= semantic_embedding_bitsize
pair_corr_o_k /= semantic_embedding_bitsize

normalizing_constant1 = 1 / np.sqrt(N*M)
normalizing_constant2 = 1 / np.sqrt(N)

mean = 0
std1 = 1 / np.sqrt(N*M)
std2 = 1 / np.sqrt(N)


flatten_corr_o_v = pair_corr_o_v.flatten()
flatten_corr_o_k = pair_corr_o_v.flatten()
flatten_corr_o_q = pair_corr_o_v.flatten()

concat_vecs = np.concatenate((flatten_corr_o_v, flatten_corr_o_k, flatten_corr_o_q))
flatten_corr_vecs = concat_vecs
flatten_corr_vecs_norm1 = concat_vecs * normalizing_constant1
flatten_corr_vecs_norm2 = concat_vecs * normalizing_constant2


# Compute histograms

density = False
hist, bin_edges = np.histogram(flatten_corr_vecs, bins=100, density=density)
width = bin_edges[1] - bin_edges[0]
# scale_factor = 0.006
scale_factor = width
hist = hist * width

x_axis = np.linspace(-1,1, 1000)
plt.title(f"N={semantic_embedding_bitsize} M={M}")
plt.stairs(hist, bin_edges, label="hist(concat(corrs))")
# plt.hist(flatten_corr_vecs_norm1, bins=100, label="hist(concat(corrs)*1/np.sqrt(N*M))", density=True)
# plt.hist(flatten_corr_vecs_norm2[:], bins=9, label="hist(concat(corrs)*1/np.sqrt(N))")
# plt.plot(x_axis, norm.pdf(x_axis, mean, std1), label="N(0, 1/sqrt(N*M))")
plt.plot(x_axis, norm.pdf(x_axis, mean, std2), label="N(mean=0, std=1/sqrt(N))")
plt.legend(fontsize="8")
plt.show()


# if num_feat_patterns == 3:
#     quad_corr_o_o = np.zeros(num_feat_patterns)
#     quad_corr_o_v = np.zeros(num_feat_patterns)
#     quad_corr_o_k = np.zeros(num_feat_patterns)
#     quad_corr_o_q = np.zeros(num_feat_patterns)
#
#     for b in range(0, num_feat_patterns):
#         for i in range(0, semantic_embedding_bitsize):
#             Wo_corr = Wo[0, i] * Wo[1, i] * Wo[2, i]
#             quad_corr_o_o[b] += Wo_corr * Wo[b, i]
#             quad_corr_o_v[b] += Wo_corr * Wv[b, i]
#             quad_corr_o_q[b] += Wo_corr * Wq[b, i]
#             quad_corr_o_k[b] += Wo_corr * Wk[b, i]
#
#     quad_corr_o_o /= semantic_embedding_bitsize
#     quad_corr_o_v /= semantic_embedding_bitsize
#     quad_corr_o_q /= semantic_embedding_bitsize
#     quad_corr_o_v /= semantic_embedding_bitsize