import numpy as np
from models.HopfieldSelfAttentionNNMFInfNPESigma import HopfieldSelfAttentionNNMFInfNPESigma
from models.Embedding import Embedding
from plotting.plotting import (plot_save_statistics, plot_save_plane, plot_save_fft, plot_save_autocorrelation,
                               plot_lyapunov_graphs, create_row_array, plot_save_statistics_1_fig)
import matplotlib.pyplot as plt
import os
import yaml
import time

def create_dir(filepath):
    plot_save_folder_path = os.path.dirname(filepath)

    # Create folder if it does not exist and we are saving the image
    if not os.path.exists(plot_save_folder_path):
        os.makedirs(plot_save_folder_path)

def save_context(context_window, folder_path_chpt, seed, num_transient_steps, max_sim_steps):
    """
    Saves the mean-field values associated to the context window
    """
    att_window, mo_window, mv_window, mq_window, mk_window, pe_window = context_window

    chpt_path = folder_path_chpt + f"/seed-{str(seed)}" + "-transient_steps-" + str(
                        num_transient_steps) + "-max_sim_steps-" + str(max_sim_steps) + ".npz"

    np.savez_compressed(chpt_path,
                        att_window=att_window,
                        mo_window=mo_window,
                        mv_window=mv_window,
                        mq_window=mq_window,
                        mk_window=mk_window,
                        pe_window=pe_window)



if __name__ == "__main__":

    # Set the variables for the experiments
    cfg_path = 'cfgs/manual_sigma_pe5/inf_0.yaml'
    # Load cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    folder_base_savepath= "results/manual_weights_ft/"

    # Instantiate vocabulary
    positional_embedding_size = 2  # Number of bits dedicated to the positional embedding
    context_size = 2 ** positional_embedding_size  # Context size
    vocab = Embedding(0, positional_embedding_size)  # Vocabulary helper class
    vocab.initialize_pos_encoder()  # Initiate some functionalities
    epsilon_pe = 0.5                                # epsilon in the paper


    # Create variables for the Hopfield Transformer (HT)
    seed = 1  # Seed for the correlations
    num_feat_patterns_se = 3                                # Number of semantic patterns
    num_feat_patterns = 4                                   # Number of patterns
    beta_list = [100]    # Different values of beta to simulate
    gamma_att = 100
    num_transient_steps = cfg["num_transient_steps"]        # Num. of transient steps
    max_sim_steps = cfg["max_sim_steps"]                    # Max. simulated steps
    saved_steps = max_sim_steps - num_transient_steps

    # Choose as initial token one of the encoded features
    ini_m_idx = 0
    seed_W = 0 # 3, 4



    compute_lyapunov = False                        # True if you want to compute the Lyapunov exponents
    save_not_plot = False                          # True -> Save; False -> Plot
    save_context_cond = False                       # If true, save context so it can be later loaded to start another execution
    show_title = True                                             # Whether to show the title on top

    for beta_o in beta_list:

        # Create seed for reproducibility
        np.random.seed(seed)

        # Create Hopfield Transformer Class
        HT = HopfieldSelfAttentionNNMFInfNPESigma(beta_o, gamma_att, num_feat_patterns=num_feat_patterns,
                                                  num_feat_patterns_se=num_feat_patterns_se,
                 positional_embedding_bitsize=positional_embedding_size, vocab=vocab, context_size=context_size,
                 max_sim_steps=max_sim_steps, min_saved_step=num_transient_steps,
                 epsilon_pe=epsilon_pe, seed_W=seed_W,
                 jacobian=True)

        # Reset/initialize the structures for saving data
        HT.reset_data()

        print(f"Simulating MF Self-Attention NN for beta {beta_o}...")
        start = time.time()
        HT.simulate(ini_m_idx, max_steps=max_sim_steps, compute_lyapunov=compute_lyapunov)
        end = time.time()
        print("Done.")
        print("Execution time = ", (end - start)/60, " minutes")

        # Plotting
        print("Plotting statistics...")
        num_plotting_steps = max_sim_steps

        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"
        folder_path = f"{folder_base_savepath}/beta{beta_o}/"

        # Create dir if it does not exist
        if save_not_plot or save_context_cond:
            create_dir(folder_path)

        if save_context_cond:
            cw = HT.get_context_window()
            save_context(cw, folder_path, seed, num_transient_steps, max_sim_steps)

        if show_title:
            title = fr"$\beta$ = {round(beta_o, 5)}, $\gamma$ = {round(gamma_att, 5)}"
        else:
            title = None
        # Select what statistic to show. One of either ["mo", "mo_se", "mv", "mq", "mk"]
        # stats_to_show = ["mo_se", "m_tilde", "m_pos", "att"]
        stats_to_show = ["mo_se", "att", "m_tilde", "m_tilde", "m_tilde_proj", "m_tilde_proj", "m_pos", "m_pos", "mo", "mv"]
        n_stats_to_show = len(stats_to_show)

        # Select format for image saving
        # image_format = ".jpeg"
        image_format = ".pdf"

        col_size = 4
        row_size = 3

        nrows = int(np.ceil(n_stats_to_show / 2))

        plot_windows = [10]  # Different plotting windows for the trajectories
        for plot_window in plot_windows:

            fig, ax = plt.subplots(nrows, 2, figsize=(col_size, row_size), constrained_layout=True)
            ax = ax.flatten()

            # Loop over the different stats if required
            for i, stat_name in zip(range(n_stats_to_show), stats_to_show):

                offset = 0  # Offset the trajectory to visit different points
                # Define the steps to show
                plot_range = [offset, offset + plot_window]  # Define the steps to plot

                if plot_range[1] >= saved_steps:
                    raise Exception("The rightmost index is greater than the number of steps.")

                rg = range(plot_range[0], plot_range[1])
                # Define path to save
                plot_save_path_traj = (folder_path + f"/traj-seed-{str(seed)}-{stat_name}" + "-transient_steps-" +
                                       str(num_transient_steps) + "-max_sim_steps-" + str(max_sim_steps)  +
                                       "-plot_window-" + str(plot_window) + image_format)
                create_dir(plot_save_path_traj)

                if stat_name == "m_tilde" or stat_name == "m_tilde_proj" or stat_name == "m_pos":
                    i_mod = i % 2

                    if i_mod == 0:
                        # o
                        selected_stats = HT.mf_statistics[stat_name][rg, 0, :]
                        stat_name += "_o"
                    else:
                        # v
                        selected_stats = HT.mf_statistics[stat_name][rg, 2, :]
                        stat_name += "_v"


                else:
                    selected_stats = HT.mf_statistics[stat_name][rg, :] 


                # Plot the trajectory
                plot_save_statistics_1_fig(ax[i], selected_stats, stat_name, selected_stats.shape[-1],
                                     len(rg), min_num_step=0,
                                     show_max_num_patterns=num_feat_patterns)

            plt.suptitle(title)
            plt.show()

            # # FFT Path
            # plot_save_path_fft = (folder_path + f"/fft-seed-{str(seed)}-{stat_name}" + "-transient_steps-" +
            #                       str(num_transient_steps) + "-max_sim_steps-" + str(max_sim_steps)  + image_format)
            #
            # # Adjust axis for the FFT if required
            # adjust_y_axis = 1.0
            # if beta == 1.266:
            #     adjust_y_axis = 0.3
            #
            # # Plot FFT
            # plot_save_fft(HT.mf_statistics[stat_name], stat_name, num_feat_patterns, saved_steps,
            #               show_max_num_patterns=num_feat_patterns,
            #               save_not_plot=save_not_plot, save_path=plot_save_path_fft, title=title,
            #               show_1_feat=show_1_feat, adjust_y_axis=adjust_y_axis)
            #
            # # Same for log FFT
            # # plot_save_path_fft_log = (folder_path + f"/log-fft-seed-{str(seed)}-{stat_name}" + "-transient_steps-" + str(num_transient_steps) + image_format)
            #
            # # plot_save_fft(HT.mf_statistics[stat_name], stat_name, num_feat_patterns, saved_steps,
            # #               show_max_num_patterns=num_feat_patterns, save_not_plot=save_not_plot,
            # #               save_path=plot_save_path_fft_log, title=title, show_1_feat=show_1_feat, log=True)
            #
            #
            # # Same for the AutoCorrelation Function
            # plot_save_path_ACF = (folder_path + f"/acf-seed-{str(seed)}-{stat_name}" + "-transient_steps-" + str(
            #             num_transient_steps) + "-max_sim_steps-" + str(max_sim_steps) + image_format)
            #
            # plot_save_autocorrelation(HT.mf_statistics[stat_name], stat_name, num_feat_patterns, saved_steps,
            #                           show_max_num_patterns=num_feat_patterns, save_not_plot=save_not_plot,
            #                           save_path=plot_save_path_ACF, title=title, show_1_feat=show_1_feat)

        print("Done.")

        plot_plane = False

        if plot_plane:

            # Define the statistics you want to plot against each other
            # In this case the feature mo with only the semantic information
            stats_to_plot = [["mo_se", "mo_se"], ["att", "att"]]
            # Define the index of the features you want to compare against each other
            feat_idx = [[0, 1], [0, 2]]

            for plot_i in range(len(stats_to_plot)):

                # Define path for saving the plane
                plot_save_path_plane = (
                        folder_path + f"/plane-seed-{str(seed)}" + "-transient_steps-" + str(num_transient_steps)
                        + "-" + stats_to_plot[plot_i][0] + "_" + str(feat_idx[plot_i][0])
                        + "-" + stats_to_plot[plot_i][1] + "_" + str(feat_idx[plot_i][1]) + image_format)

                # Set larger dots for the periodic trajectory
                larger_dots = False
                dpi = None

                # Create figure
                ncols = 1
                fig, ax = create_row_array(ncols, dpi)

                # Load statistics
                stat_results_beta_0 = HT.mf_statistics[stats_to_plot[plot_i][0]][:, feat_idx[plot_i][0]]
                stat_results_beta_1 = HT.mf_statistics[stats_to_plot[plot_i][0]][:, feat_idx[plot_i][1]]
                # Plot plane
                plot_save_plane(stat_results_beta_0,
                                stat_results_beta_1, max_sim_steps - num_transient_steps, feat_idx[plot_i], ax,
                                tag_names=stats_to_plot[plot_i], title=title, larger_dots=larger_dots)

                if save_not_plot:
                    fig.savefig(plot_save_path_plane, bbox_inches='tight')
                else:
                    plt.show()
                plt.close()

            lowres_lya = False
            image_format_lya = image_format
            if lowres_lya:
                image_format_lya = ".jpg"

            if compute_lyapunov:
                # Reorder in descending order, filter out components associated to Positional Encoding rotations (last components)
                sorted_S = np.sort(HT.S[:HT.num_feat_patterns * HT.context_size])[::-1]

                print("Sorted Lyapunov exponents in descencing order", sorted_S)
                plot_save_path_lya = (
                        folder_path + f"/lyapunov-{str(seed)}" + "-transient_steps-" + str(
                    num_transient_steps) + "-max_sim_steps-" + str(max_sim_steps) + image_format_lya)
                # Plot lyapunov related statistics

                plot_lyapunov_graphs(HT.S_i_sum, cfg, beta_o,
                                     save_not_plot=save_not_plot, save_path=plot_save_path_lya)

                print("Inf flag")
                print(HT.S_inf_flag)