import matplotlib.pyplot as plt
from utils import feat_name_to_latex
import numpy as np

# LaTeX macros
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
font = {'size': 24, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 14})


def plot_bifurcation_diagram(mo_results_beta_list, beta_list, num_feat_patterns, save_path, num_transient_steps,
                             feat_name, show_max_num_patterns=None, save_not_plot=True):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    nrows = (num_feat_patterns + 1) // 2

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(24, 4), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, 4 * nrows), constrained_layout=True)

    latex_str = feat_name_to_latex(feat_name)

    for feat in range(0, num_feat_patterns):

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        elif num_feat_patterns == 3:
            local_ax = ax[feat % 3]
        else:
            local_ax = ax[row, feat % 2]

        feat_Y_values = []
        feat_X_values = []

        for b_idx in range(0, len(beta_list)):
            unique_values_feat = mo_results_beta_list[b_idx][num_transient_steps:, feat]
            beta_values_feat = np.ones(len(unique_values_feat)) * beta_list[b_idx]

            feat_Y_values.extend(unique_values_feat)
            feat_X_values.extend(beta_values_feat)

        local_ax.plot(feat_X_values, feat_Y_values, ls='', marker='.', ms='0.05')
        if feat_name != "att" and beta_list[-1] > 3.5:
            local_ax.set_ylim(-1, 1)


        local_ax.set_xlim(beta_list[0], beta_list[-1])

        if num_feat_patterns==3:
            local_ax.set_xlabel(r"$\beta$")
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(r"$\beta$")

        local_ax.set_ylabel(fr"${latex_str}_{{{feat},t}}$")
        # local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    # fig.suptitle(f"Bifurcation_diagram {feat_name}")
    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_save_statistics(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                         save_not_plot=False, save_path=None):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    nrows = (num_feat_patterns + 1) // 2

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(24, 4), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, 4 * nrows), constrained_layout=True)

    num_plotting_steps_arange = np.arange(num_plotting_steps)

    latex_str = feat_name_to_latex(stat_name)

    for feat in range(0, num_feat_patterns):

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        elif num_feat_patterns == 3:
            local_ax = ax[feat % 3]
        else:
            local_ax = ax[row, feat % 2]

        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, feat])

        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$t$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$t$")

        local_ax.set_ylabel(fr"${latex_str}_{{{feat},t}}$")

        # local_ax.legend()

    # fig.tight_layout(pad=0.1)
    # fig.suptitle(f"Evolution of {stat_name}")

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_save_fft(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                  save_not_plot=False, save_path=None):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    nrows = (num_feat_patterns + 1) // 2

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(24, 4), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, 4 * nrows), constrained_layout=True)

    num_plotting_steps_arange = np.arange(num_plotting_steps)

    latex_str = feat_name_to_latex(stat_name)

    for feat in range(0, num_feat_patterns):

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        elif num_feat_patterns == 3:
            local_ax = ax[feat % 3]
        else:
            local_ax = ax[row, feat % 2]

        stat_fft = np.fft.rfft(stat1[:num_plotting_steps, feat]).real
        freqs = np.fft.rfftfreq(num_plotting_steps)
        local_ax.plot(freqs, stat_fft)
        # local_ax.specgram(stat1[:num_plotting_steps, feat], Fs=1)

        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$Hz$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$Hz$")

        local_ax.set_ylabel(fr"Mag. of ${latex_str}_{{{feat},t}}$")

        # local_ax.legend()

    # fig.tight_layout(pad=0.1)
    # fig.suptitle(f"Evolution of {stat_name}")

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()
