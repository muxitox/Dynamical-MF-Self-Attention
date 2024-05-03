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
    # vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    # 2.2 with seed 2 gives some cycles
    # beta = 0.3779 # We'll find the nearest beta in the defined range
    # beta 1.26 1.266 1.28 1.40
    beta = 1.266 / np.sqrt(3)
    print(beta)

    # 2 patts: 2.4, 2.45


    num_feat_patterns = 3
    num_transient_steps = 1000  # 0 if we want to show the trajectory since the beginning
    saved_steps = 20000
    max_sim_steps = num_transient_steps + saved_steps
    # num_transient_steps_traj = max_sim_steps - num_transient_steps - 250  # 0 if we want to show the trajectory since the beginning

    plot_window = 1000
    offset = 1000 + plot_window
    plot_range = [saved_steps - offset - 1, saved_steps - offset + plot_window - 1]


    correlations_from_weights = 3
    pe_mode = 0
    se_per_contribution = tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)
    # se_per_contribution = 0.98

    N_normalization = 99
    # multiplier_o = N_normalization + positional_embedding_size
    # multiplier_att = N_normalization + positional_embedding_size
    scaling_o = 1
    scaling_att = 10

    beta_o = beta
    # beta_att = beta * np.sqrt(num_feat_patterns)
    beta_att = betagit
    # beta_att = beta
    print(beta_o, beta_att)

    # normalize_weights_str_att = "np.sqrt(N)*M"
    # normalize_weights_str_o = "np.sqrt(N)*M"
    # normalize_weights_str_att = "N**2"
    normalize_weights_str_att = "N**2"
    normalize_weights_str_o = "N"
    compute_inf_normalization = True
    reorder_weights = False
    ini_token_from_w = 1
    save_not_plot = False

    num_ini_tokens = 10
    # Select initial token with seed 0
    np.random.seed(0)
    ini_tokens_list = np.random.randint(2, size=(num_ini_tokens, tentative_semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    ini_token_idx = 0
    if ini_token_from_w == 0:
        x0 = ini_tokens_list[ini_token_idx, :]

    # Interesting seeds: 18, 26
    # seed_list = range(30, 60)

    seed_list = [1]
    for seed in seed_list:

        # Create seed for reproducibility
        np.random.seed(seed)

        HT = HopfieldTransformerInfN(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                     positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                     context_size=context_size, max_sim_steps=max_sim_steps,
                                     min_saved_step=num_transient_steps,
                                     normalize_weights_str_att=normalize_weights_str_att,
                                     normalize_weights_str_o=normalize_weights_str_o,
                                     reorder_weights=reorder_weights,
                                     correlations_from_weights=correlations_from_weights,
                                     semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                     se_per_contribution=se_per_contribution, pe_mode=pe_mode,
                                     compute_inf_normalization=compute_inf_normalization,
                                     N_normalization=N_normalization,
                                     scaling_o=scaling_o,
                                     scaling_att=scaling_att)

        if ini_token_from_w == 1:
            x0 = HT.Wo[ini_token_idx]
            x0[-positional_embedding_size:] = -1  # Initialize position to -1

        HT.reset_data()

        print(HT.even_corr_o_o)
        print(HT.even_corr_o_v)
        print(HT.even_corr_o_q)
        print(HT.even_corr_o_k)


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

        num_segments_corrs = 3
        gaussian_scale_name_str= ""
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"
        beta_list = np.linspace(0.35, 0.8, 1000)
        reverse_betas_str = ""
        keep_context = False
        folder_path = ("results/infN-correlations_from_weights-" + str(correlations_from_weights)
                       + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                       + str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                       + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str
                       + "-reorder_weights-" + str(int(reorder_weights))
                       + "-num_segments_corrs-" + str(num_segments_corrs) + "-pe_mode-" + str(pe_mode)
                       + gaussian_scale_name_str + "/max_sim_steps-" + str(max_sim_steps)
                       + save_non_transient_str + "-context_size-" + str(context_size)
                       + "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1])
                       + "-num_betas-" + str(len(beta_list)) + f"{reverse_betas_str}-keep_context-"
                       + str(int(keep_context)))


        print(folder_path)

        title = (f"SEED={seed} BETA={beta} SE_PER={se_per_contribution} CONTEXT={context_size}  "
                 f" NUM_TRANSIENT={num_transient_steps} MODE={correlations_from_weights}")
        # title = fr"$\beta={beta}$"


        stats_to_show = ["mo_se"]
        image_format = ".jpeg"

        for stat_name in stats_to_show:
            plot_save_path_traj = (
                        folder_path + f"/indiv_traj/latex/seed-{str(seed)}/{stat_name}/beta-{beta}-ini_token_idx-" +
                        str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

            plot_save_path_fft = (
                        folder_path + f"/indiv_traj/latex/seed-{str(seed)}/{stat_name}/fft-beta-{beta}-ini_token_idx-" +
                        str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

            create_dir(plot_save_path_traj)
            create_dir(plot_save_path_fft)

            rounded = np.round(HT.mf_statistics[stat_name], decimals=5)
            print(stat_name, len(np.unique(rounded[:,0])),
                  len(np.unique(rounded[:,1])),
                  len(np.unique(rounded[:,2])),
                  len(np.unique(rounded, axis=0)))

            rg = range(plot_range[0], plot_range[1])
            plot_save_statistics(HT.mf_statistics[stat_name][rg, :], stat_name, num_feat_patterns,
                                 len(rg), min_num_step=num_transient_steps + plot_range[0],
                                 show_max_num_patterns=num_feat_patterns,
                                 save_not_plot=save_not_plot, save_path=plot_save_path_traj, title=title,
                                 plot_hilbert=True)

            plot_save_fft(HT.mf_statistics[stat_name], stat_name, num_feat_patterns,
                          saved_steps,
                          show_max_num_patterns=num_feat_patterns,
                          save_not_plot=save_not_plot, save_path=plot_save_path_fft)

            # plot_save_fft(HT.mf_statistics[stat_name][num_transient_steps_traj:,:], stat_name, num_feat_patterns,
            #               max_sim_steps - num_transient_steps - num_transient_steps_traj,
            #               show_max_num_patterns=num_feat_patterns, save_path=plot_save_path_fft,
            #               save_not_plot=save_not_plot)

        print("Done.")


        # 3 feats
        stats_to_plot = [["mo", "mk", "mv"], ["mo", "mk", "mv"]]
        if num_feat_patterns == 3:
            feat_idx = [[0, 1, 2], [1, 0, 1]]
        elif num_feat_patterns == 2:
            feat_idx = [[0, 0, 0], [1, 1, 1]]
        elif num_feat_patterns == 1:
            stats_to_plot = [["mo", "mk", "mk"], ["mv", "mq", "mv"]]
            feat_idx = [[0, 0, 0], [0, 0, 0]]

        plot_save_path_plane = (folder_path + f"/indiv_traj/latex/seed-{str(seed)}/planes"
                                + f"/plane-beta-{beta}-ini_token_idx-" +
                                str(ini_token_idx) + "-transient_steps-" + str(num_transient_steps) + image_format)

        create_dir(plot_save_path_plane)

        stat_results_beta_list_0 = [HT.mf_statistics[stats_to_plot[0][0]], HT.mf_statistics[stats_to_plot[0][1]],
                                    HT.mf_statistics[stats_to_plot[0][2]]]
        stat_results_beta_list_1 = [HT.mf_statistics[stats_to_plot[1][0]], HT.mf_statistics[stats_to_plot[1][1]],
                                    HT.mf_statistics[stats_to_plot[1][2]]]
        # feats_reorder = np.roll(np.arange(num_feat_patterns), -1)
        plot_save_plane(stat_results_beta_list_0,
                        stat_results_beta_list_1, max_sim_steps - num_transient_steps, feat_idx,
                        tag_names=stats_to_plot, save_path=plot_save_path_plane, save_not_plot=save_not_plot)

        if num_feat_patterns == 3:

            # stats_to_plot = ["mo_se", "mk", "mv"]
            # stat_results_beta_list = [HT.mf_statistics[stats_to_plot[0]], HT.mf_statistics[stats_to_plot[1]],
            #                             HT.mf_statistics[stats_to_plot[2]]]

            stats_to_plot = ["mo_se"]
            stat_results_beta_list = [HT.mf_statistics[stats_to_plot[0]]]

            plot_save_3Dplane(stat_results_beta_list, max_sim_steps - num_transient_steps,
                              tag_names=stats_to_plot, beta=beta, save_path=plot_save_path_plane,
                              save_not_plot=save_not_plot)
