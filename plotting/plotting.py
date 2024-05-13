import matplotlib.pyplot as plt
from utils import feat_name_to_latex
import numpy as np
from scipy.signal import hilbert, chirp
import matplotlib

cmap = matplotlib.colormaps["inferno_r"]
colors = []
for i_color in range(9):
    colors += [cmap(i_color / 8)]
colors += [(1.0, 1.0, 1.0, 1.0)]  # White



# LaTeX macros
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
font = {'size': 24, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 14})

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

            local_ax.plot(beta_values_feat, unique_values_feat, c=colors[2], ls='', marker='.', ms='0.05')


        if feat_name != "att" and x_list[-1] > 3.5:
            local_ax.set_ylim(-1, 1)


        local_ax.set_xlim(x_list[0], x_list[-1])

        if num_feat_patterns==3:
            local_ax.set_xlabel(x_label)
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(x_label)

        local_ax.set_ylabel(fr"${latex_str}_{{{feat+1},t}}$")
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
    row_size = 4

    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(24, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, row_size * nrows), constrained_layout=True)

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

            unique_values_feat = np.unique(np.round(unique_values_feat, decimals=3))
            unique_values_feat_filtered = np.unique(np.round(unique_values_feat_filtered, decimals=3))

            beta_values_feat = np.ones(len(unique_values_feat)) * x_list[b_idx]
            beta_values_feat_filtered = np.ones(len(unique_values_feat_filtered)) * x_list[b_idx]

            local_ax.plot(beta_values_feat, unique_values_feat, c=colors[2], ls='', marker='.', ms='0.05')
            local_ax.plot(beta_values_feat_filtered, unique_values_feat_filtered, c=colors[0], ls='', marker='.', ms='0.5')

        if feat_name != "att" and x_list[-1] > 3.5:
            local_ax.set_ylim(-1, 1)

        local_ax.set_xlim(x_list[0], x_list[-1])

        if num_feat_patterns == 3:
            local_ax.set_xlabel(x_label)
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(x_label)

        local_ax.set_ylabel(fr"${latex_str}_{{{feat+1},t}}$")
        # local_ax.legend(loc="upper center")

    # fig.tight_layout(pad=0.1)
    if title is not None:
        fig.suptitle(title)

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()


def filter_y_values_by_0_plane(results_y_list, feat, filter_idx, filtering_range):
    filtering_values = results_y_list[:, filter_idx]
    zero_intersect = np.where(np.logical_and(filtering_values >= -filtering_range,
                                             filtering_values <= filtering_range))
    return results_y_list[zero_intersect, feat]


def plot_filtered_bifurcation_diagram_par(filter_idx, x_list, num_feat_patterns,
                                          save_path, num_transient_steps, feat_name,
                                          folder_path, seed, ini_token_idx, ini_token_mode_str,
                                          filtering_range=0.05,
                                          show_max_num_patterns=None, save_not_plot=True, title=None,
                                          is_beta=True, min_bidx=0, show_1_feat=None,
                                          filter_periodic=100):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:
        feat_list = [show_1_feat]
        num_feat_patterns = 1

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

    for feat_id in range(0, num_feat_patterns):
        feat = feat_list[feat_id]

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        elif num_feat_patterns == 3:
            local_ax = ax[feat % 3]
        else:
            local_ax = ax[row, feat % 2]

        for idx in range(len(x_list)):
            b_idx = min_bidx + idx
            stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                               + str(ini_token_idx) + ini_token_mode_str + "-beta_idx-" + str(b_idx)
                               + ".npz")

            # Load data
            data = np.load(stats_data_path)
            results_y_list = data[f"{feat_name}_results_beta"]

            unique_values_feat = results_y_list[num_transient_steps:, feat]
            unique_values_feat_filtered = filter_y_values_by_0_plane(results_y_list[num_transient_steps:], feat,
                                                                     filter_idx, filtering_range)
            dec = 3
            unique_values_feat = np.unique(np.round(unique_values_feat, decimals=dec))
            unique_values_feat_filtered = np.unique(np.round(unique_values_feat_filtered, decimals=dec))

            beta_values_feat = np.ones(len(unique_values_feat)) * x_list[b_idx]
            beta_values_feat_filtered = np.ones(len(unique_values_feat_filtered)) * x_list[b_idx]

            local_ax.plot(beta_values_feat, unique_values_feat, c=colors[2], ls='', marker='.', ms='0.05')

            if len(unique_values_feat) < filter_periodic:
                local_ax.plot(beta_values_feat, unique_values_feat, c=colors[1], ls='',
                              marker='.', ms='0.5')  # Periodic
            else:
                local_ax.plot(beta_values_feat_filtered, unique_values_feat_filtered, c=colors[0], ls='',
                              marker='.', ms='0.5')  # Other

        local_ax.plot(0.1, 0, 'o', c=colors[1], label="Periodic")
        local_ax.plot(0.1, 0, 'o', c=colors[0], label="Other")
        local_ax.plot(0.1, 0, 'o', c="w")


        if feat_name != "att" and x_list[-1] > 3.5:
            local_ax.set_ylim(-1, 1)

        local_ax.set_xlim(x_list[0], x_list[-1])
        local_ax.set_xlim(x_list[0], x_list[-1])


        if num_feat_patterns == 3:
            local_ax.set_xlabel(x_label)
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(x_label)

        local_ax.set_ylabel(fr"${latex_str}_{{{feat+1},t}}$")
        local_ax.legend(loc="upper left")

    # fig.tight_layout(pad=0.1)
    if title is not None:
        fig.suptitle(title)

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()


def compute_max_min(x_list, folder_path, seed, ini_token_idx, ini_token_mode_str, min_bidx, feat_name):
    max_y = - np.inf
    min_y = np.inf
    for idx in range(len(x_list)):
        b_idx = min_bidx + idx
        stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                           + str(ini_token_idx) + ini_token_mode_str + "-beta_idx-" + str(b_idx)
                           + ".npz")

        # Load data
        data = np.load(stats_data_path)
        results_y_list = data[f"{feat_name}_results_beta"]
        local_min = np.min(results_y_list)
        local_max = np.max(results_y_list)
        if local_min < min_y:
            min_y = local_min
        if local_max > max_y:
            max_y = local_max

    return min_y, max_y

def create_imshow_array(x_list, folder_path, seed, ini_token_idx, ini_token_mode_str, min_bidx, feat_name,
                        num_transient_steps, feat, bins, y_resolution):

    im_array = np.ones((y_resolution, len(x_list), 4)) * colors[-1]
    for idx in range(len(x_list)):
        b_idx = min_bidx + idx
        stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                           + str(ini_token_idx) + ini_token_mode_str + "-beta_idx-" + str(b_idx)
                           + ".npz")

        # Load data
        data = np.load(stats_data_path)
        results_y_list = data[f"{feat_name}_results_beta"]

        values_feat = results_y_list[num_transient_steps:, feat]
        inds = np.unique(np.digitize(values_feat, bins, right=False)).astype(int) - 1

        im_array[inds, idx] = colors[3]  # All points

    return im_array

def get_filtered_values_by_beta(idx, folder_path, seed, ini_token_idx, ini_token_mode_str, min_bidx, feat_name,
                        num_transient_steps, feat, bins, filter_idx, filtering_range):
    b_idx = min_bidx + idx
    stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                       + str(ini_token_idx) + ini_token_mode_str + "-beta_idx-" + str(b_idx)
                       + ".npz")

    # Load data
    data = np.load(stats_data_path)
    results_y_list = data[f"{feat_name}_results_beta"]

    values_feat_filtered = filter_y_values_by_0_plane(results_y_list[num_transient_steps:], feat,
                                                      filter_idx, filtering_range)
    inds_filtered = np.unique(np.digitize(values_feat_filtered, bins, right=True)).astype(int) - 1

    values_feat_unfiltered = results_y_list[num_transient_steps:, feat]
    inds_unfiltered = np.unique(np.digitize(values_feat_unfiltered, bins, right=True)).astype(int) - 1

    unique_len = len(inds_unfiltered)
    return bins[inds_filtered], bins[inds_unfiltered], unique_len


def plot_filtered_bifurcation_diagram_par_imshow(filter_idx, x_list, num_feat_patterns,
                                          save_path, num_transient_steps, feat_name,
                                          folder_path, seed, ini_token_idx, ini_token_mode_str,
                                          filtering_range=0.05,
                                          show_max_num_patterns=None, save_not_plot=True, title=None,
                                          is_beta=True, min_bidx=0, show_1_feat=None,
                                          filter_periodic=80):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:
        feat_list = [show_1_feat]
        num_feat_patterns = 1

    nrows = (num_feat_patterns + 1) // 2

    col_size = 5
    row_size = 3
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(col_size, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(col_size*num_feat_patterns, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(col_size*2, row_size * nrows), constrained_layout=True)

    latex_str = feat_name_to_latex(feat_name)
    x_label = r'$\beta$'
    if not is_beta:
        x_label = r'$SE\%$'

    for feat_id in range(0, num_feat_patterns):
        feat = feat_list[feat_id]

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        elif num_feat_patterns == 3:
            local_ax = ax[feat % 3]
        else:
            local_ax = ax[row, feat % 2]


        # Compute min max range to define the im_array properly
        min_y, max_y = compute_max_min(x_list, folder_path, seed, ini_token_idx, ini_token_mode_str, min_bidx,
                                       feat_name)

        y_resolution = int(len(x_list)*2 + 1)
        bins_size = (max_y - min_y) / y_resolution
        bins = np.arange(y_resolution) * bins_size - ((max_y - min_y) / 2)

        # 1 - First plot all the quantized values for all betas
        im_array = create_imshow_array(x_list, folder_path, seed, ini_token_idx, ini_token_mode_str, min_bidx,
                                       feat_name, num_transient_steps, feat, bins, y_resolution)

        local_ax.imshow(im_array, interpolation=None, extent=[x_list[0], x_list[-1], -min_y, min_y])
        local_ax.set_aspect("auto")

        # 2- Then, apply a plane intersection filter
        # Step 1 and 2 are done separately to avoid memory issues
        for idx in range(len(x_list)):
            values_feat_filtered_quantized, rounded_feat_results_y_list, unique_len = (
                get_filtered_values_by_beta(idx, folder_path, seed, ini_token_idx, ini_token_mode_str, min_bidx,
                                            feat_name, num_transient_steps, feat, bins, filter_idx, filtering_range))

            beta_values_feat = np.ones(len(rounded_feat_results_y_list)) * x_list[idx]
            beta_values_quantized_feat = np.ones(len(values_feat_filtered_quantized)) * x_list[idx]


            if unique_len < filter_periodic:
                local_ax.plot(beta_values_feat, rounded_feat_results_y_list, c=colors[0], ls='',
                              marker='.', ms='0.008')  # Periodic
            else:
                local_ax.plot(beta_values_quantized_feat, values_feat_filtered_quantized, c=colors[-3], ls='',
                              marker='.', ms='0.008')  # Other

        local_ax.plot(4, 0, 'o', c=colors[0], label="Periodic")
        local_ax.plot(4, 0, 'o', c=colors[-3], label="Other")
        local_ax.set_xlim([x_list[0], x_list[-1]])
        local_ax.set_xlabel(x_label)
        local_ax.set_ylabel(fr"${latex_str}_{{{feat+1},t}}$")
        local_ax.legend(loc="upper left")

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

    row_size = 3
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(24, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, row_size * nrows), constrained_layout=True)

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

        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$t$")
        elif feat > num_feat_patterns-3:
            local_ax.set_xlabel(r"$t$")

        local_ax.set_ylabel(fr"${latex_str}_{{{feat+1},t}}$")
        local_ax.legend()

    # fig.tight_layout(pad=0.1)
    fig.suptitle(f"Evolution of {stat_name}{additional_msg}")
    plt.show()

def plot_save_statistics(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                         save_not_plot=False, save_path=None, min_num_step=0, title=None, plot_hilbert=False,
                         show_1_feat=None):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)


    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:
        feat_list = [show_1_feat]
        num_feat_patterns = 1

    nrows = (num_feat_patterns + 1) // 2

    col_size = 9
    row_size = 3
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(col_size*num_feat_patterns, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(col_size*num_feat_patterns, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(col_size*2, row_size * nrows), constrained_layout=True)

    num_plotting_steps_arange = np.arange(num_plotting_steps) + min_num_step

    latex_str = feat_name_to_latex(stat_name)

    for feat_id in range(0, num_feat_patterns):

        feat = feat_list[feat_id]
        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        elif num_feat_patterns == 3:
            local_ax = ax[feat % 3]
        else:
            local_ax = ax[row, feat % 2]


        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, feat], c="k", label="Signal")
        if plot_hilbert:
            analytic_signal = hilbert(stat1[:num_plotting_steps, feat])
            local_ax.plot(num_plotting_steps_arange, np.abs(analytic_signal), c=colors[0], alpha=0.5, label="Hilbert tr")


        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$t$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$t$")

        local_ax.set_ylabel(fr"${latex_str}_{{{feat+1},t}}$")

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
                    save_not_plot=False, save_path=None, tag_names=[], title=None, lowres=False):

    # Plot show_max_num_patterns subfigures if defined

    ncols = len(stats1)
    if ncols != len(stats2):
        raise Exception("The length of the input stats does not coincide")

    dpi = None
    ms = 0.4
    if lowres:
        dpi = 15
        ms = 0.7

    fig, ax = plt.subplots(1, ncols, figsize=(8*ncols, 8), constrained_layout=True, dpi=dpi)

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

        local_ax.plot(stats1[feat][:num_plotting_steps, feat_idx[0][feat]],
                      stats2[feat][:num_plotting_steps,feat_idx[1][feat]], '.', c="k", ms=ms)

        local_ax.set_xlabel(rf"${latex_strs[0][feat]}_{feat_idx[0][feat]+1}(t)$")
        local_ax.set_ylabel(rf"${latex_strs[1][feat]}_{feat_idx[1][feat]+1}(t)$")

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
                  save_not_plot=False, save_path=None, mode="sq", title=None, show_1_feat=None, log=False):

    # mode may be sq or real
    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:
        feat_list = [show_1_feat]
        num_feat_patterns = 1

    nrows = (num_feat_patterns + 1) // 2
    row_size = 3
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(24, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, row_size * nrows), constrained_layout=True)

    latex_str = feat_name_to_latex(stat_name)

    for feat_id in range(0, num_feat_patterns):

        feat = feat_list[feat_id]

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

        log_str = ""
        if log:
            log_str = "\log "
            stat_fft = np.log(stat_fft)

        local_ax.plot(freqs, stat_fft, c="k")
        # local_ax.specgram(stat1[:num_plotting_steps, feat], Fs=1)

        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$Hz$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$Hz$")

        if mode == "real":
            local_ax.set_ylabel(fr"${log_str}re(\mathcal{{F}} f_{{ {latex_str}_{feat+1} }})$")
        else:
            local_ax.set_ylabel(fr"${log_str}|\mathcal{{F}}|^2 f_{{ {latex_str}_{feat+1}  }}$")


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


def plot_save_autocorrelation(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                  save_not_plot=False, save_path=None, title=None, show_1_feat=None, plotting_window=100):

    # mode may be sq or real
    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:
        feat_list = [show_1_feat]
        num_feat_patterns = 1

    nrows = (num_feat_patterns + 1) // 2
    row_size = 3
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(24, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(16, row_size * nrows), constrained_layout=True)

    latex_str = feat_name_to_latex(stat_name)

    for feat_id in range(0, num_feat_patterns):

        feat = feat_list[feat_id]

        row = feat // 2
        if num_feat_patterns == 1:
            local_ax = ax
        elif num_feat_patterns == 2:
            local_ax = ax[feat % 2]
        elif num_feat_patterns == 3:
            local_ax = ax[feat % 3]
        else:
            local_ax = ax[row, feat % 2]

        stat_autocorr = np.correlate(stat1[:num_plotting_steps, feat], stat1[:num_plotting_steps, feat], mode="full")

        middle_index = len(stat1[:num_plotting_steps, feat]) - 1

        stat_autocorr_window = stat_autocorr[middle_index:middle_index+plotting_window]
        x_vals = np.arange(len(stat_autocorr_window))

        local_ax.plot(x_vals, stat_autocorr_window, c="k")

        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$t - \tau$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$t - \tau$")

        local_ax.set_ylabel(fr"ACF ${latex_str}_{feat+1}$")

        local_ax.set_xlim(x_vals[0], x_vals[-1])

        if title is not None:
            fig.suptitle(title)

    if save_not_plot:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()
