import matplotlib.pyplot as plt
from utils import feat_name_to_latex
import numpy as np
from scipy.signal import hilbert, chirp

# LaTeX macros
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# font = {'size': 24, 'family': 'serif', 'serif': ['latin modern roman']}
# plt.rc('font', **font)
# plt.rc('legend', **{'fontsize': 14})


def plot_bifurcation_diagram(results_y_list, x_list, num_feat_patterns, save_path, num_transient_steps,
                             feat_name, show_max_num_patterns=None, save_not_plot=True, title=None, is_beta=True):

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
    x_label = r'$\beta$'
    if not is_beta:
        x_label = r'$SE\%$'

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

        for b_idx in range(0, len(x_list)):
            unique_values_feat = results_y_list[b_idx][num_transient_steps:, feat]
            beta_values_feat = np.ones(len(unique_values_feat)) * x_list[b_idx]

            local_ax.plot(beta_values_feat, unique_values_feat, c='tab:blue', ls='', marker='.', ms='0.05')


        if feat_name != "att" and x_list[-1] > 3.5:
            local_ax.set_ylim(-1, 1)


        local_ax.set_xlim(x_list[0], x_list[-1])

        if num_feat_patterns==3:
            local_ax.set_xlabel(x_label)
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(x_label)

        local_ax.set_ylabel(fr"${latex_str}_{{{feat},t}}$")
        # local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    if title is not None:
        fig.suptitle(title)

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_filtered_bifurcation_diagram(results_y_list, filtering_variable, filter_idx, x_list, num_feat_patterns,
                                      save_path, num_transient_steps, feat_name, filtering_range=0.05,
                                      show_max_num_patterns=None, save_not_plot=True, title=None, is_beta=True):

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
    x_label = r'$\beta$'
    if not is_beta:
        x_label = r'$SE\%$'

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

        for b_idx in range(0, len(x_list)):
            filtering_values = filtering_variable[b_idx][num_transient_steps:, filter_idx]
            zero_intersect = np.where(np.logical_and(filtering_values >= -filtering_range,
                                                     filtering_values <= filtering_range))
            unique_values_feat = results_y_list[b_idx][num_transient_steps:, feat]
            unique_values_feat_filtered = unique_values_feat[zero_intersect]

            beta_values_feat = np.ones(len(unique_values_feat_filtered)) * x_list[b_idx]

            local_ax.plot(beta_values_feat, unique_values_feat_filtered, c='tab:blue', ls='', marker='.', ms='0.5')

        if feat_name != "att" and x_list[-1] > 3.5:
            local_ax.set_ylim(-1, 1)

        local_ax.set_xlim(x_list[0], x_list[-1])

        if num_feat_patterns == 3:
            local_ax.set_xlabel(x_label)
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(x_label)

        local_ax.set_ylabel(fr"${latex_str}_{{{feat},t}}$")
        # local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    if title is not None:
        fig.suptitle(title)

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()



def plot_2_statistics(stat1, stat2, stat_name, num_feat_patterns, num_plotting_steps, label_tag,
                      show_max_num_patterns=None, additional_msg=""):

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

        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, feat], label=label_tag[0])
        local_ax.plot(num_plotting_steps_arange, stat2[:num_plotting_steps, feat], '--', label=label_tag[1])

        if num_feat_patterns==3:
            local_ax.set_xlabel(r"$t$")
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(r"$t$")

        local_ax.set_ylabel(fr"${latex_str}_{{{feat},t}}$")
        local_ax.legend()

    # fig.tight_layout(pad=0.1)
    fig.suptitle(f"Evolution of {stat_name}{additional_msg}")
    plt.show()

def plot_save_statistics(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                         save_not_plot=False, save_path=None, min_num_step=0, title=None, plot_hilbert=False):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    nrows = (num_feat_patterns + 1) // 2

    col_size = 9

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(col_size*num_feat_patterns, 4), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(col_size*num_feat_patterns, 4), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(col_size*2, 4 * nrows), constrained_layout=True)

    num_plotting_steps_arange = np.arange(num_plotting_steps) + min_num_step

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


        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, feat], label="Signal")
        if plot_hilbert:
            analytic_signal = hilbert(stat1[:num_plotting_steps, feat])
            local_ax.plot(num_plotting_steps_arange, np.abs(analytic_signal), alpha=0.5, label="Hilbert tr")


        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$t$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$t$")

        local_ax.set_ylabel(fr"${latex_str}_{{{feat},t}}$")

        local_ax.set_xlim(num_plotting_steps_arange[0], num_plotting_steps_arange[-1])

        if plot_hilbert:
            local_ax.legend()

    # fig.tight_layout(pad=0.1)
    if title is not None:
        fig.suptitle(title)

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_save_plane(stats1, stats2, num_plotting_steps, feat_idx,
                    save_not_plot=False, save_path=None, tag_names=[], title=None):

    # Plot show_max_num_patterns subfigures if defined

    ncols = len(stats1)
    if ncols != len(stats2):
        raise Exception("The length of the input stats does not coincide")

    fig, ax = plt.subplots(1, ncols, figsize=(8*ncols, 8), constrained_layout=True)

    # num_plotting_steps_arange = np.arange(num_plotting_steps)

    # Convert stat names to latex style
    latex_strs = []
    for col_name in tag_names:
        strs = []
        for tag in col_name:
            strs.append(feat_name_to_latex(tag))
        latex_strs.append(strs)

    for feat in range(0, ncols):

        if ncols == 1:
            local_ax = ax
        else:
            local_ax = ax[feat % ncols]

        local_ax.plot(stats1[feat][:num_plotting_steps, feat_idx[0][feat]], stats2[feat][:num_plotting_steps, feat_idx[1][feat]], '.', ms='0.4')

        local_ax.set_xlabel(rf"${latex_strs[0][feat]}_{feat_idx[0][feat]}(t)$")
        local_ax.set_ylabel(rf"${latex_strs[1][feat]}_{feat_idx[1][feat]}(t)$" )

        # local_ax.legend()

    # fig.tight_layout(pad=0.1)
    # fig.suptitle(f"Evolution of {stat_name}")
    if title is not None:
        fig.suptitle(title)

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_save_3Dplane(stats1, num_plotting_steps,
                    save_not_plot=False, save_path=None, tag_names=[], beta=None):

    # Plot show_max_num_patterns subfigures if defined

    ncols = len(stats1)
    # if ncols != len(stats2):
    #     raise Exception("The length of the input stats does not coincide")

    fig = plt.figure(figsize=(8*ncols, 4), constrained_layout=True)

    # fig.add_subplot()
    # num_plotting_steps_arange = np.arange(num_plotting_steps)

    # Convert stat names to latex style
    latex_strs = []

    for tag in tag_names:
        latex_strs.append(feat_name_to_latex(tag))

    for stat_id in range(0, ncols):

        # if ncols == 1:
        #     local_ax = ax
        # else:
        #     local_ax = ax[stat_id % ncols]

        local_ax = fig.add_subplot(1, ncols, 1, projection='3d')

        local_ax.scatter(stats1[stat_id][:num_plotting_steps, 0], stats1[stat_id][:num_plotting_steps, 1],
                         stats1[stat_id][:num_plotting_steps, 2])

        local_ax.set_xlabel(rf"${latex_strs[stat_id]}_0(t)$")
        local_ax.set_ylabel(rf"${latex_strs[stat_id]}_1(t)$" )
        # local_ax.set_zlabel(rf"${latex_strs[stat_id]}_2(t)$" )

        # local_ax.legend()

    # fig.tight_layout(pad=0.1)
    # fig.suptitle(f"Evolution of {stat_name}")
    fig.suptitle(rf"$\beta$ = {beta}")


    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_save_fft(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                  save_not_plot=False, save_path=None, mode="sq", title=None):

    # mode may be sq or real

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

        stat_fft = np.fft.rfft(stat1[:num_plotting_steps, feat])
        if mode == "real":
            stat_fft = stat_fft.real

        else:
            stat_fft = stat_fft.real**2 + stat_fft.imag**2

        freqs = np.fft.rfftfreq(num_plotting_steps)
        local_ax.plot(freqs, stat_fft)
        # local_ax.specgram(stat1[:num_plotting_steps, feat], Fs=1)

        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$Hz$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$Hz$")

        if mode == "real":
            local_ax.set_ylabel(fr"re(Mag.) of ${latex_str}_{{{feat},t}}$")
        else:
            local_ax.set_ylabel(fr"Mag.$^2$ of ${latex_str}_{{{feat},t}}$")


        local_ax.set_xlim(freqs[0], freqs[-1])

        if title is not None:
            fig.suptitle(title)

        # local_ax.legend()

    # fig.tight_layout(pad=0.1)
    # fig.suptitle(f"Evolution of {stat_name}")

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()
