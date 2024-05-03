import numpy as np
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from models.HopfieldTransformerPE import Embedding
from plotting.plotting import plot_save_statistics, plot_save_plane, plot_save_fft, plot_save_3Dplane
import os

def create_dir(filepath):
    plot_save_folder_path = os.path.dirname(filepath)

    # Create folder if it does not exist and we are saving the image
    if save_not_plot and (not os.path.exists(plot_save_folder_path)):
        os.makedirs(plot_save_folder_path)

if __name__ == "__main__":

    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size
    embedding_size = tentative_semantic_embedding_size + positional_embedding_size
    vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size)
    vocab.initialize_pos_encoder()

    # Create variables for the Hopfield Transformer (HT)
    seed = 1
    beta_list = [1.26, 1.266, 1.28, 1.4]
    # beta_list = [1.266]
    num_feat_patterns = 3
    num_transient_steps = 100000  # 0 if we want to show the trajectory since the beginning
    saved_steps = 20000
    max_sim_steps = num_transient_steps + saved_steps

    correlations_from_weights = 3
    pe_mode = 0
    # se_per_contribution = tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)
    se_per_contribution = 0.98
    scaling_o = 1
    scaling_att = 100

    normalize_weights_str_att = "N**2"
    normalize_weights_str_o = "N"
    compute_inf_normalization = True
    ini_token_idx = 0
    ini_token_from_w = 1
    save_not_plot = True

    for beta in beta_list:

        # Create seed for reproducibility
        np.random.seed(seed)

        HT = HopfieldTransformerInfN(beta, beta, num_feat_patterns=num_feat_patterns,
                                     positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                     context_size=context_size, max_sim_steps=max_sim_steps,
                                     min_saved_step=num_transient_steps,
                                     normalize_weights_str_att=normalize_weights_str_att,
                                     normalize_weights_str_o=normalize_weights_str_o,
                                     correlations_from_weights=correlations_from_weights,
                                     semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                     se_per_contribution=se_per_contribution, pe_mode=pe_mode,
                                     compute_inf_normalization=compute_inf_normalization,
                                     scaling_o=scaling_o,
                                     scaling_att=scaling_att)

        # Set initial token from one of the output features
        x0 = HT.Wo[ini_token_idx]
        x0[-positional_embedding_size:] = -1  # Initialize position to -1

        HT.reset_data()  # Reset the structures for saving data


        print("Simulating MF Transformer...")
        HT.simulate_mf(x0, max_steps=max_sim_steps)
        print("Done.")

        # Plotting
        print("Plotting statistics...")
        num_plotting_steps = max_sim_steps

        if normalize_weights_str_o == normalize_weights_str_att:
            normalize_weights_name_str = "-normalize_weights-" + normalize_weights_str_att
        else:
            normalize_weights_name_str = ("-normalize_weights_att-" + normalize_weights_str_att +
                                          "-normalize_weights_o-" + normalize_weights_str_o)


        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"
        folder_path = f"results_eurnips/beta{beta}"
        create_dir(folder_path)

        title = fr"$\beta$ = {beta}"
        stats_to_show = ["mo_se"]
        image_format = ".jpeg"

        for stat_name in stats_to_show:

            plot_windows = [250, 5000]
            for plot_window in plot_windows:
                offset = 1000 + plot_window
                plot_range = [saved_steps - offset - 1, saved_steps - offset + plot_window - 1]
                plot_save_path_traj = (folder_path + f"/traj-seed-{str(seed)}-{stat_name}" + "-ini_token-" +
                                       str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) +
                                       "-plot_window-" + str(plot_window) + image_format)
                create_dir(plot_save_path_traj)

                rg = range(plot_range[0], plot_range[1])
                plot_save_statistics(HT.mf_statistics[stat_name][rg, :], stat_name, num_feat_patterns,
                                     len(rg), min_num_step=num_transient_steps + plot_range[0],
                                     show_max_num_patterns=num_feat_patterns,
                                     save_not_plot=save_not_plot, save_path=plot_save_path_traj, title=title,
                                     plot_hilbert=False)

            plot_save_path_fft = (folder_path + f"/fft-seed-{str(seed)}-{stat_name}" + "-ini_token-" +
                                  str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

            plot_save_fft(HT.mf_statistics[stat_name], stat_name, num_feat_patterns,
                          saved_steps,
                          show_max_num_patterns=num_feat_patterns,
                          save_not_plot=save_not_plot, save_path=plot_save_path_fft, title=title)

        print("Done.")

        # 1 feat
        stats_to_plot = [["mo_se"], ["mo_se"]]
        feat_idx = [[0], [1]]

        plot_save_path_plane = (
                folder_path + f"/plane-seed-{str(seed)}" + "-ini_token-" +
                str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

        stat_results_beta_list_0 = [HT.mf_statistics[stats_to_plot[feat_idx[0][0]][0]]]
        stat_results_beta_list_1 = [HT.mf_statistics[stats_to_plot[feat_idx[1][0]][0]]]
        plot_save_plane(stat_results_beta_list_0,
                        stat_results_beta_list_1, max_sim_steps - num_transient_steps, feat_idx,
                        tag_names=stats_to_plot, beta=beta, save_path=plot_save_path_plane, save_not_plot=save_not_plot)


        # # 3 feats
        # stats_to_plot = [["mo", "mk", "mv"], ["mo", "mk", "mv"]]
        # if num_feat_patterns == 3:
        #     feat_idx = [[0, 1, 2], [1, 0, 1]]
        # elif num_feat_patterns == 2:
        #     feat_idx = [[0, 0, 0], [1, 1, 1]]
        # elif num_feat_patterns == 1:
        #     stats_to_plot = [["mo", "mk", "mk"], ["mv", "mq", "mv"]]
        #     feat_idx = [[0, 0, 0], [0, 0, 0]]
        #
        # plot_save_path_plane = (
        #         folder_path + f"/plane-seed-{str(seed)}" + "-ini_token-" +
        #         str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)
        #
        # stat_results_beta_list_0 = [HT.mf_statistics[stats_to_plot[0][0]], HT.mf_statistics[stats_to_plot[0][1]],
        #                             HT.mf_statistics[stats_to_plot[0][2]]]
        # stat_results_beta_list_1 = [HT.mf_statistics[stats_to_plot[1][0]], HT.mf_statistics[stats_to_plot[1][1]],
        #                             HT.mf_statistics[stats_to_plot[1][2]]]
        # plot_save_plane(stat_results_beta_list_0,
        #                 stat_results_beta_list_1, max_sim_steps - num_transient_steps, feat_idx,
        #                 tag_names=stats_to_plot,  beta=beta, save_path=plot_save_path_plane, save_not_plot=save_not_plot)
