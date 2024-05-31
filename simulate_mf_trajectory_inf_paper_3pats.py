import numpy as np
from models.HopfieldTransformerMFInfNPE import HopfieldTransformerMFInfNPE
from models.Embedding import Embedding
from plotting.plotting import plot_save_statistics, plot_save_plane, plot_save_fft, plot_save_autocorrelation
import os

def create_dir(filepath):
    plot_save_folder_path = os.path.dirname(filepath)

    # Create folder if it does not exist and we are saving the image
    if save_not_plot and (not os.path.exists(plot_save_folder_path)):
        os.makedirs(plot_save_folder_path)

def load_context(chpt_path):

    cw = np.load(chpt_path)

    return cw['mv_window'], cw['mq_window'], cw['mk_window'], cw['att_window']

if __name__ == "__main__":

    # Set the variables for the experiments

    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99  # Variable to set the size of the matrices from which to compute the corrs
    positional_embedding_size = 2  # Number of bits dedicated to the positional embedding
    context_size = 2 ** positional_embedding_size  # Context size
    embedding_size = tentative_semantic_embedding_size + positional_embedding_size  # Total size of the "tentative" embedding
    vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size) # Vocabulary helper class
    vocab.initialize_pos_encoder()  # Initiate some functionalities

    # Create variables for the Hopfield Transformer (HT)
    seed = 1  # Seed for the correlations
    beta_list = [1.255, 1.26427, 1.266, 1.27, 1.28, 1.4]    # Different values of beta to simulate
    scaling_o = 1  # Not scaled
    beta_att = 2.2
    scaling_att = 100                                       # Beta_att * scaling_att make gamma
    num_feat_patterns = 3                                   # Number of patterns
    num_transient_steps = 100000                            # Num. of transient steps
    saved_steps = 20000                                     # Num. of steps to save
    max_sim_steps = num_transient_steps + saved_steps       # Max. simulated steps

    correlations_from_weights = 3                           # Variable for choosing how to set the correlations
                                                            # If = 3, we compute them from segments as in the paper
    pe_mode = 0                                             # Choose how to initialize the PE. Set it randomly.
    epsilon_pe = 0.02                                       # epsilon in the paper

    normalize_weights_str_att = "N**2*np.sqrt(M)"           # U in the paper
    normalize_weights_str_o = "N"                           # Normalization in the output
    compute_inf_normalization = True                        # Deal with normalization constraint in infinity

    save_not_plot = True                                    # True -> Save; False -> Plot
    show_title = True                                       # Whether to show the title on top

    # Load checkpoint attention values
    chpt_path = "neurips_chpt/beta_idx-4000_window_chpt_zoom.npz"
    mv_window_chpt, mq_window_chpt, mk_window_chpt, att_window_chpt = load_context(chpt_path)

    for beta in beta_list:

        # Create seed for reproducibility
        np.random.seed(seed)

        # Create Hopfield Transformer Class
        HT = HopfieldTransformerMFInfNPE(beta, beta_att, num_feat_patterns=num_feat_patterns,
                                         positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                         context_size=context_size, max_sim_steps=max_sim_steps,
                                         min_saved_step=num_transient_steps,
                                         normalize_weights_str_att=normalize_weights_str_att,
                                         normalize_weights_str_o=normalize_weights_str_o,
                                         correlations_from_weights=correlations_from_weights,
                                         semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                         epsilon_pe=epsilon_pe, pe_mode=pe_mode,
                                         compute_inf_normalization=compute_inf_normalization,
                                         scaling_o=scaling_o,
                                         scaling_att=scaling_att,
                                         N_normalization=9999)

        # Reset/initialize the structures for saving data
        HT.reset_data()

        print(f"Simulating MF Transformer for beta {beta}...")

        # Set context window to the checkpoint values
        HT.set_context_window(mv_window_chpt, mq_window_chpt, mk_window_chpt, att_window_chpt)
        HT.simulate_mf_from_context(max_steps=max_sim_steps)
        print("Done.")

        # Plotting
        print("Plotting statistics...")
        num_plotting_steps = max_sim_steps

        # Create some strings to define the paths for saving the plots
        if normalize_weights_str_o == normalize_weights_str_att:
            normalize_weights_name_str = "-normalize_weights-" + normalize_weights_str_att
        else:
            normalize_weights_name_str = ("-normalize_weights_att-" + normalize_weights_str_att +
                                          "-normalize_weights_o-" + normalize_weights_str_o)

        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"
        folder_path = f"results_neurips/beta{beta}"

        # Create dir if it does not exist
        create_dir(folder_path)

        if show_title:
            title = fr"$\beta$ = {round(beta, 5)}"
        else:
            title = None
        # Select what statistic to show. One of either ["mo", "mo_se", "mv", "mq", "mk"]
        stats_to_show = ["mo_se"]

        # Select format for image saving
        # image_format = ".jpeg"
        image_format = ".pdf"


        # Loop over the different stats if required
        for stat_name in stats_to_show:
            show_1_feat = 0  # Defines that it's only going to show 1 feature and what's its index
            plot_windows = [250, 350, 5000]  # Different plotting windows for the trajectories
            for plot_window in plot_windows:
                offset = 900 + plot_window  # Offset the trajectory to visit different points
                # Define the steps to show
                plot_range = [saved_steps - offset - 1, saved_steps - offset + plot_window - 1]  # Define the steps to plot
                rg = range(plot_range[0], plot_range[1])
                # Define path to save
                plot_save_path_traj = (folder_path + f"/traj-seed-{str(seed)}-{stat_name}" + "-transient_steps-" +
                                       str(num_transient_steps) +
                                       "-plot_window-" + str(plot_window) + image_format)
                create_dir(plot_save_path_traj)

                # Plot the trajectory
                plot_save_statistics(HT.mf_statistics[stat_name][rg, :], stat_name, num_feat_patterns,
                                     len(rg), min_num_step=0,
                                     show_max_num_patterns=num_feat_patterns,
                                     save_not_plot=save_not_plot, save_path=plot_save_path_traj, title=title,
                                     plot_hilbert=False, show_1_feat=show_1_feat)

            # FFT Path
            plot_save_path_fft = (folder_path + f"/fft-seed-{str(seed)}-{stat_name}" + "-transient_steps-" +
                                  str(num_transient_steps) + image_format)

            # Adjust axis for the FFT if required
            adjust_y_axis = 1.0
            if beta == 1.266:
                adjust_y_axis = 0.3

            # Plot FFT
            plot_save_fft(HT.mf_statistics[stat_name], stat_name, num_feat_patterns, saved_steps,
                          show_max_num_patterns=num_feat_patterns,
                          save_not_plot=save_not_plot, save_path=plot_save_path_fft, title=title,
                          show_1_feat=show_1_feat, adjust_y_axis=adjust_y_axis)

            # Same for log FFT
            plot_save_path_fft_log = (folder_path + f"/log-fft-seed-{str(seed)}-{stat_name}" + "-transient_steps-" + str(num_transient_steps) + image_format)

            plot_save_fft(HT.mf_statistics[stat_name], stat_name, num_feat_patterns, saved_steps,
                          show_max_num_patterns=num_feat_patterns, save_not_plot=save_not_plot,
                          save_path=plot_save_path_fft_log, title=title, show_1_feat=show_1_feat, log=True)


            # Same for the AutoCorrelation Function
            plot_save_path_ACF = (folder_path + f"/acf-seed-{str(seed)}-{stat_name}" + "-transient_steps-" + str(
                        num_transient_steps) + image_format)

            plot_save_autocorrelation(HT.mf_statistics[stat_name], stat_name, num_feat_patterns, saved_steps,
                                      show_max_num_patterns=num_feat_patterns, save_not_plot=save_not_plot,
                                      save_path=plot_save_path_ACF, title=title, show_1_feat=show_1_feat)

        print("Done.")

        # Define the statistics you want to plot against each other
        # In this case the feature mo with only the semantic information
        stats_to_plot = [["mo_se"], ["mo_se"]]
        # Define the index of the features you want to compare against each other
        feat_idx = [[0], [1]]

        # Define path for saving the plane
        plot_save_path_plane = (
                folder_path + f"/plane-seed-{str(seed)}" + "-transient_steps-" + str(num_transient_steps) + image_format)

        # Set larger dots for the periodic trajectory
        larger_dots = False
        if beta == 1.27:
            larger_dots = True

        # Load statistics
        stat_results_beta_list_0 = [HT.mf_statistics[stats_to_plot[feat_idx[0][0]][0]]]
        stat_results_beta_list_1 = [HT.mf_statistics[stats_to_plot[feat_idx[1][0]][0]]]
        # Plot plane
        plot_save_plane(stat_results_beta_list_0,
                        stat_results_beta_list_1, max_sim_steps - num_transient_steps, feat_idx,
                        tag_names=stats_to_plot, save_path=plot_save_path_plane, save_not_plot=save_not_plot,
                        title=title, larger_dots=larger_dots)
