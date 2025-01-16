import platform
import matplotlib.pyplot as plt
from utils import feat_name_to_latex
import numpy as np
from scipy.signal import hilbert, chirp
import matplotlib

cmap = matplotlib.colormaps["inferno_r"]
colors = []
intensity_multiplier = 1.1
for i_color in range(9):
    color = cmap(i_color / 8)
    color_intensity = min(color[-2] * intensity_multiplier, 1.0)  # Increase intensity
    color = list(color)
    color[-2] = color_intensity
    colors += [tuple(color)]
colors += [(1.0, 1.0, 1.0, 1.0)]  # White


# Avoid loading latex in HPC (not installed there)
if "Ubuntu" in platform.version():
    # LaTeX macros
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    font = {'size': 34, 'family': 'serif', 'serif': ['latin modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 14})

rowsize = 8
colsize = 8

def create_row_array(ncols, dpi):
    fig, ax = plt.subplots(1, ncols, figsize=(colsize * ncols, rowsize), constrained_layout=True, dpi=dpi)

    return fig, ax


def create_imshow_array(x_list, folder_path, min_bidx, feat_name, num_transient_steps, feat, y_resolution, max_y):

    # Creates array for imshow
    white = [(1.0, 1.0, 1.0, 1.0)]
    im_array = np.ones((y_resolution, len(x_list), 4)) * white # By default, white color

    for idx in range(len(x_list)):
        b_idx = min_bidx + idx
        stats_data_path = (folder_path + "/stats" + "/beta_idx-" + str(b_idx)
                           + ".npz")

        # Load data
        data = np.load(stats_data_path)
        results_y_list = data[f"{feat_name}_results_beta"]
        values_feat = results_y_list[num_transient_steps:, feat]
        # Quantization into y_resolution values
        inds = np.unique((y_resolution * (values_feat/max_y + 1) / 2).astype(int)) - 1

        im_array[inds, idx] = cmap(0.09)  # All points. Sets the color 0.09 from the colormap

    return im_array

def plot_bifurcation_diagram(feat_to_plot, x_list, num_transient_steps, feat_name, stats_folder_path, local_ax,
                             min_y, max_y,
                             x_label=r'$\beta$', min_bidx=0):
    """
    Plots a bifurcation diagram
    :param feat_to_plot: id of the feature for which to plot the bifurcation diagram
    :param x_list: x domain values. Usually betas.
    :param num_transient_steps: Number of transient steps
    :param feat_name: Feature name. For plotting purposes.
    :param stats_folder_path: Folder path in which we saved the results.
    :param local_ax: fig axes in which we plot the bifurcation diagram.
    :param min_y: min value of `feat_to_plot`.
    :param max_y: max absolute value of `feat_to_plot` in the whole bifurcation diagram.
    :param x_label: label associated with the x axis.
    :param min_bidx: min index of the x_values list to be plotted.
    """

    abs_max_y = max(abs(min_y), abs(max_y))

    y_resolution_im = 7001

    # 1 - First plot all the quantized values for all x values
    im_array = create_imshow_array(x_list, stats_folder_path, min_bidx,
                                   feat_name, num_transient_steps, feat_to_plot, y_resolution_im, abs_max_y)

    local_ax.imshow(im_array, cmap=cmap, interpolation=None, extent=[x_list[0], x_list[-1], -min_y, min_y],
                    rasterized=True, aspect="auto", alpha=1)

    if len(x_list) > 20: # Otherwise the diagram is too strech
        local_ax.set_aspect("auto")

    # Set x label
    local_ax.set_xlabel(x_label)

    # Get the feature name nice to plot
    latex_str = feat_name_to_latex(feat_name)
    # Rotate y label
    kwargs = {}
    kwargs["rotation"] = "horizontal"
    kwargs["verticalalignment"] = "center"
    labelpad = 34
    local_ax.set_ylabel(fr"${latex_str}_{{{feat_to_plot + 1},t}}$", labelpad=labelpad, **kwargs)



def filter_bifurcation_diagram_by_couting(x_list, x_idx_to_filter, values_feat_filtered_quantized, values_feat_quantized,
                                       unique_len, local_ax, filter_periodic=80):
    """
    Plots a bifurcation diagram
    :param x_list: x domain values. Usually betas.
    :param x_idx_to_filter: x value to filter
    :param values_feat_filtered_quantized: y values filtered after quantization
    :param values_feat_quantized: y values quantized
    :param unique_len: number of unique y values
    :param local_ax:
    :param filter_periodic: Number of steps by which we consider some beta has a periodic behavior.
    """

    # 2- Then, apply a plane intersection filter
    # Step 1 and 2 are done separately to avoid memory issues


    x_values_feat = np.ones(len(values_feat_quantized)) * x_list[x_idx_to_filter]
    x_values_quantized_feat = np.ones(len(values_feat_filtered_quantized)) * x_list[x_idx_to_filter]

    ms = 0.08  # Markersize

    if unique_len < filter_periodic:
        local_ax.plot(x_values_feat, values_feat_quantized, c=cmap(3 / 3), ls='',
                      marker=',', ms=ms, rasterized=True)  # Periodic
    elif unique_len < 2500:
        local_ax.plot(x_values_quantized_feat, values_feat_filtered_quantized, c=cmap(1.5 / 4), ls='',
                      marker='.', ms=ms, rasterized=True)  # Quasi
    else:
        local_ax.plot(x_values_quantized_feat, values_feat_filtered_quantized, c=cmap(2.5 / 4), ls='',
                      marker='.', ms=ms, rasterized=True)  # Chaos


    # Only define legend the first time we plot an x value
    if x_idx_to_filter == 0:
        # Plot tiny points outside the map for the legend.
        local_ax.plot(20, 0, 'o', c=cmap(3 / 3), label="periodic")
        local_ax.plot(20, 0, 'o', c=cmap(1.5 / 4), label="quasi-periodic")
        local_ax.plot(20, 0, 'o', c=cmap(2.5 / 4), label="chaotic")
        local_ax.set_xlim([x_list[0], x_list[-1]])

        # Plot only the legend in the zoomed in version of the beta's bifurcation diagram
        if x_list[-1] < 3:
            local_ax.legend(loc="upper left")

    # Return values in case some function needs it for further analysis
    return values_feat_quantized, values_feat_filtered_quantized


def plot_phase_diagram(unique_matrix, unique_filtered_matrix, att_values_list, out_values_list, save_path):

    num_betas_att = len(att_values_list)
    num_betas_out = len(out_values_list)
    im_array = np.ones((num_betas_att, num_betas_out, 4)) * colors[-1]  # By default, white color

    for i in range(num_betas_att):
        for j in range(num_betas_out):
            if unique_matrix[i][j] <= 80:
                # periodic
                im_array[i][j] = cmap(3/3)
                # pass
            elif unique_matrix[i][j] <= 625:
                # quasi
                im_array[i][j] = cmap(1.5/4)
            else:
                # chaos
                im_array[i][j] = cmap(2.5/4)

    plt.imshow(im_array, cmap=cmap, interpolation=None, extent=(att_values_list[0], att_values_list[-1],
                                                                out_values_list[0], out_values_list[-1]),
                    rasterized=True, aspect="auto", alpha=1)

    # Plot tiny points outside the map for the legend.
    plt.plot(20, 0, 'o', c=cmap(3 / 3), label="periodic")
    plt.plot(20, 0, 'o', c=cmap(1.5 / 4), label="quasi-periodic")
    plt.plot(20, 0, 'o', c=cmap(2.5 / 4), label="chaotic")

    kwargs = {}
    kwargs["rotation"] = "horizontal"
    kwargs["verticalalignment"] = "center"
    labelpad = 15

    plt.xlim([out_values_list[0], out_values_list[-1]],)

    plt.ylabel(rf"$\gamma$", labelpad=labelpad, **kwargs)
    plt.xlabel(rf"$\beta$")
    plt.tight_layout()
    plt.legend(loc="upper left", fontsize=12)
    plt.savefig(save_path)
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

    plt.close()


def plot_save_statistics(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                         save_not_plot=False, save_path=None, min_num_step=0, title=None, plot_hilbert=False,
                         show_1_feat=None):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    # Define the feats to plot
    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:  # If show_1_feat, only plot that feature
        feat_list = [show_1_feat]
        num_feat_patterns = 1


    # Set params for figsize. Create fig and axes.
    nrows = (num_feat_patterns + 1) // 2

    col_size = 8
    row_size = 3
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(col_size*num_feat_patterns, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(col_size*num_feat_patterns, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(col_size*2, row_size * nrows), constrained_layout=True)

    # X domain
    num_plotting_steps_arange = np.arange(num_plotting_steps) + min_num_step

    # Strings for the labels
    latex_str = feat_name_to_latex(stat_name)

    # For each feature
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

        # Plot trajectory
        local_ax.plot(num_plotting_steps_arange, stat1[:num_plotting_steps, feat], c="k", label="Signal")
        if plot_hilbert:  # If set, plot Hilbert transform
            analytic_signal = hilbert(stat1[:num_plotting_steps, feat])
            local_ax.plot(num_plotting_steps_arange, np.abs(analytic_signal), c=colors[0], alpha=0.5, label="Hilbert tr")

        # Labelling
        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$t$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$t$")

        # Rotate labels
        kwargs = {}
        kwargs["rotation"] = "horizontal"
        kwargs["verticalalignment"] = "center"
        labelpad = 34
        local_ax.set_ylabel(fr"${latex_str}_{{{feat+1},t}}$", labelpad=labelpad, **kwargs)

        local_ax.set_xlim(num_plotting_steps_arange[0], num_plotting_steps_arange[-1])

        # If Hilbert tr. is requested, set the legend.
        if plot_hilbert:
            local_ax.legend()

    # fig.tight_layout(pad=0.1)
    if title is not None:
        local_ax.set_title(title)

    if save_not_plot:
        fig.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_save_plane(stats1, stats2, num_plotting_steps, feat_idxs, local_ax,
                    tag_names=[], title=None, lowres=False, larger_dots=False):

    """
    :param local_ax: axes from a subplots function must be provided to plot the diagram onto.

    """
    # Plot show_max_num_patterns subfigures if defined

    ncols = len(stats1)
    if ncols != len(stats2):
        raise Exception("The length of the input stats does not coincide")

    # Set markersize depending on the set up
    ms = 0.4
    if lowres:
        ms = 0.7
    elif larger_dots:  # If not a lowres execution and larger_dots is set, print them bigger
        ms = 10


    # Convert stat names to latex style
    latex_strs = []
    for tag in tag_names:
        latex_strs.append(feat_name_to_latex(tag))


    # Plot trajectories against each other
    local_ax.plot(stats1[:num_plotting_steps],
                  stats2[:num_plotting_steps], '.', c="k", ms=ms, rasterized=True)

    # x label
    local_ax.set_xlabel(rf"${latex_strs[0]}_{{{feat_idxs[0]+1},t}}$")

    # Rotate y label
    kwargs = {}
    kwargs["rotation"] = "horizontal"
    kwargs["verticalalignment"] = "center"
    labelpad = 25
    local_ax.set_ylabel(rf"${latex_strs[1]}_{{{feat_idxs[1]+1},t}}$", labelpad=labelpad, **kwargs)

    if title is not None:
        local_ax.set_title(title)


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
        local_ax.set_ylabel(rf"${latex_strs[stat_id]}_1(t)$")

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
                  save_not_plot=False, save_path=None, mode="sq", title=None, show_1_feat=None, log=False,
                  adjust_y_axis=1.0):

    # mode may be "sq" or "real"
    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:  # If show_1_feat, only plot that feature
        feat_list = [show_1_feat]
        num_feat_patterns = 1

    nrows = (num_feat_patterns + 1) // 2
    row_size = 3
    col_size = 8
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(col_size, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(col_size * 3, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(col_size * 2, row_size * nrows), constrained_layout=True)

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

        # FFT Transform
        stat_fft = np.fft.rfft(stat1[:num_plotting_steps, feat])
        if mode == "real":
            stat_fft = stat_fft.real

        else:
            stat_fft = stat_fft.real**2 + stat_fft.imag**2

        # Normalize by length
        stat_fft /= len(stat1[:num_plotting_steps, feat])

        # Get freqs
        freqs = np.fft.rfftfreq(num_plotting_steps)

        # If logarithm is requested, plot it in the legend
        log_str = ""
        if log:
            log_str = "\log "
            stat_fft = np.log(stat_fft)

        # Plot
        local_ax.plot(freqs, stat_fft, c="k")

        if num_feat_patterns == 3:
            local_ax.set_xlabel(rf"$f$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(rf"$f$")

        kwargs = {}
        kwargs["rotation"] = "horizontal"
        kwargs["verticalalignment"] = "center"
        labelpad = 34
        if mode == "real":
            local_ax.set_ylabel(fr"${log_str}re(\mathcal{{F}}_f)$", labelpad=labelpad, **kwargs)
        else:
            local_ax.set_ylabel(fr"${log_str}|\mathcal{{F}}_f|^2$", labelpad=labelpad, **kwargs)

        # Change xlim to better view the highest frequencies
        local_ax.set_xlim(freqs[0]-0.005, freqs[-1]+0.005)

        # Adjust y axis if requested
        local_ax.set_ylim(min(stat_fft), max(stat_fft) * adjust_y_axis)

        if title is not None:
            local_ax.set_title(title)

        # local_ax.legend()

    # fig.tight_layout(pad=0.1)
    # fig.suptitle(f"Evolution of {stat_name}")

    if save_not_plot:
        fig.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_save_autocorrelation(stat1, stat_name, num_feat_patterns, num_plotting_steps, show_max_num_patterns=None,
                  save_not_plot=False, save_path=None, title=None, show_1_feat=None, plotting_window=100):

    # Plot show_max_num_patterns subfigures if defined
    if (show_max_num_patterns is not None):
        num_feat_patterns = min(num_feat_patterns, show_max_num_patterns)

    feat_list = np.arange(num_feat_patterns)
    if show_1_feat is not None:  # If show_1_feat, only plot that feature
        feat_list = [show_1_feat]
        num_feat_patterns = 1

    # Create figsize
    nrows = (num_feat_patterns + 1) // 2
    row_size = 3
    col_size = 8
    if num_feat_patterns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(col_size, row_size), constrained_layout=True)
    elif num_feat_patterns == 3:
        fig, ax = plt.subplots(1, 3, figsize=(3*col_size, row_size), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(2*col_size, row_size * nrows), constrained_layout=True)

    # latex_str = feat_name_to_latex(stat_name)

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

        # Compute autocorrelation
        stat_autocorr = np.correlate(stat1[:num_plotting_steps, feat], stat1[:num_plotting_steps, feat], mode="full")

        middle_index = len(stat1[:num_plotting_steps, feat]) - 1

        # Select which values to plot
        stat_autocorr_window = stat_autocorr[middle_index:middle_index+plotting_window]
        x_vals = np.arange(len(stat_autocorr_window))

        # Plot values
        local_ax.plot(x_vals, stat_autocorr_window, c="k")

        #  X label
        if num_feat_patterns == 3:
            local_ax.set_xlabel(r"$\ell$")
        elif feat > num_feat_patterns - 3:
            local_ax.set_xlabel(r"$\ell$")

        # Rotate labels
        kwargs = {}
        kwargs["rotation"] = "horizontal"
        kwargs["verticalalignment"] = "center"
        labelpad = 34
        local_ax.set_ylabel(fr"$R_\ell$", labelpad=labelpad, **kwargs)

        local_ax.set_xlim(x_vals[0], x_vals[-1])

        if title is not None:
            local_ax.set_title(title)

    if save_not_plot:
        fig.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_lyapunov_graphs(S_i_sum, cfg, beta, save_not_plot=False, save_path=None, lowres=False):

    M = cfg["num_feat_patterns"]
    pe_bit_size = cfg["positional_embedding_size"]
    context_size = cfg["context_size"]

    dpi = None
    if lowres:
        dpi = 15

    ncols = 3
    # Create figure
    fig, ax = plt.subplots(1, ncols, figsize=(8*ncols, 8), constrained_layout=True, dpi=dpi)

    ax[0].plot(S_i_sum[:, :M])
    ax[0].set_title("Main feats")

    ax[1].plot(S_i_sum[-1000:, :M])
    ax[1].set_title("Main feats. End zoom.")

    ax[2].plot(S_i_sum[:, M:-(pe_bit_size * context_size)], )
    ax[2].set_title("Remaining feats")

    plt.suptitle(rf'$\beta$={beta}')

    if save_not_plot:
        fig.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plot_bifurcation_lyapunov(x_list, num_feat_patterns, context_size, folder_path, save_basepath, save_not_plot=True, title=None,
                  min_bidx=0):
    """
    Plots statistics related with the Lyapunov exponents of a bifurcation diagram
    """


    # Process Lyapunov exponents and mark errors

    lyapunov_size = num_feat_patterns * context_size

    S_array = np.zeros((len(x_list), lyapunov_size))
    S_array_inf = np.zeros((len(x_list), lyapunov_size))

    count_failures = 0


    for idx in range(1, len(x_list)):
        b_idx = min_bidx + idx
        stats_data_path = (folder_path + "/stats" + "/beta_idx-" + str(b_idx) + ".npz")
        # Load data
        data = np.load(stats_data_path)
        # Load only variables not associated with the copying of Positional Encoding values
        S_array[idx] = data["S"][:lyapunov_size]
        S_array_inf[idx] = S_inf_flag = data["S_inf_flag"][:lyapunov_size]


        if True in S_inf_flag:
            print(x_list[b_idx], "Fallo exponentes")
            print(S_inf_flag)
            print()
            count_failures += 1


    S_array_inf_any = np.any(S_array_inf, axis=0)
    print("Affected idxs", S_array_inf_any)
    print("Num failures", count_failures)

    # First plot evolution of lyapunov exponents

    valid_S = S_array[:,np.logical_not(S_array_inf_any)]
    num_valid_dims = valid_S.shape[1]

    col_size = 5
    row_size = 4
    dpi = 250
    fig, ax = plt.subplots(2, 1, figsize=(col_size, row_size), constrained_layout=True, dpi=dpi)

    ax[0].plot(x_list[1:], valid_S[1:,:num_feat_patterns])
    ax[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax[0].set_ylabel(r"$\lambda^{1-3}$")

    ax[1].plot(x_list[1:], valid_S[1:,num_feat_patterns:])
    ax[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax[1].set_xlabel(r"$\beta$")
    ax[0].set_ylabel(r"$\lambda^{other}$")

    if save_not_plot:
        fig.savefig(save_basepath + "/lyapunov_evolution.png")
    else:
        fig.show()

    plt.close(fig)

    col_size = 8
    row_size = 6
    dpi = 250
    # Then plot histogram of the 3 main features
    fig, ax = plt.subplots(num_feat_patterns, 1, figsize=(col_size, row_size), dpi=dpi, constrained_layout=True)
    flat_ax = ax.ravel()

    for i in range(num_feat_patterns):
        flat_ax[i].hist(S_array[:, i], bins=200)
        flat_ax[i].set_xlabel(rf"$\lambda_{i}$")
        flat_ax[i].set_ylabel("Freqs.")

    if save_not_plot:
        fig.savefig(save_basepath + "/hist_1.png")
    else:
        fig.show()
    plt.close(fig)


    # Then plot the hist of the remaining features
    num_other_feats = num_valid_dims - num_feat_patterns
    row_size = 16
    fig, ax = plt.subplots(num_other_feats, 1, figsize=(col_size, row_size), dpi=dpi)
    flat_ax = ax.ravel()

    for i in range(num_other_feats):
        flat_ax[i].hist(S_array[1:, i + num_feat_patterns], bins=200)
        flat_ax[i].set_xlabel(rf"$\lambda_{i+num_feat_patterns}$")
        flat_ax[i].set_ylabel("Freqs.")

    plt.tight_layout()

    if save_not_plot:
        fig.savefig(save_basepath + "/hist_2.png")
    else:
        fig.show()
    plt.close(fig)


    fig = plt.figure(figsize=(8, 8))
    plt.plot(S_array[1:, 0], S_array[1:, 1], '.', c="k", rasterized=True)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel(r"$S_1$")
    plt.ylabel(r"$S_2$")
    # plt.xlim([min(S_array[1:, 0]), max(S_array[1:, 0])])
    # plt.ylim([min(S_array[1:, 1]), max(S_array[1:, 1])])
    fig.tight_layout()
    if save_not_plot:
        fig.savefig(save_basepath + "/plane.png")
    else:
        fig.show()
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    local_ax = fig.add_subplot(1, 1, 1, projection='3d')
    local_ax.scatter(S_array[1:, 0], S_array[1:, 1], x_list[1:])
    local_ax.set_xlabel(r"$\lambda_1$")
    local_ax.set_ylabel(r"$\lambda_2$")
    local_ax.set_zlabel(r"$\beta$")
    if save_not_plot:
        fig.savefig(save_basepath + "/3Dplane.png")
    else:
        fig.show()
    plt.close(fig)

