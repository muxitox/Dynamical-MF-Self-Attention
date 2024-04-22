import numpy as np
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from models.HopfieldTransformerPE import Embedding
from plotting.plotting import plot_save_statistics, plot_save_plane, plot_save_fft
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
    # vocab.initialize()

    # Create variables for the Hopfield Transformer (HT)
    # 2.2 with seed 2 gives some cycles
    beta = 2.2  # We'll find the nearest beta in the defined range

    # 2 patts: 2.4, 2.45

    beta_o = beta
    beta_att = beta

    num_feat_patterns = 3
    num_transient_steps = 100000  # 0 if we want to show the trajectory since the beginning
    max_sim_steps = num_transient_steps + 200000
    num_transient_steps_traj = max_sim_steps - num_transient_steps - 250  # 0 if we want to show the trajectory since the beginning

    correlations_from_weights = 3
    pe_mode = 0
    se_per_contribution = tentative_semantic_embedding_size / (tentative_semantic_embedding_size + positional_embedding_size)

    N_normalization = 9999
    normalize_weights_str = "N*np.sqrt(M)"
    compute_inf_normalization = True
    reorder_weights = False

    save_not_plot = False

    num_ini_tokens = 1
    # Select initial token with seed 0
    np.random.seed(0)
    ini_tokens_list = np.random.randint(2, size=(num_ini_tokens, tentative_semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    ini_token_idx = 0
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
                                     min_saved_step=num_transient_steps, normalize_weights_str=normalize_weights_str,
                                     reorder_weights=reorder_weights,
                                     correlations_from_weights=correlations_from_weights,
                                     semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                     se_per_contribution=se_per_contribution, pe_mode=pe_mode,
                                     compute_inf_normalization=compute_inf_normalization,
                                     N_normalization=N_normalization)

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


        num_segments_corrs = 3
        gaussian_scale_name_str= ""
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"
        beta_list = np.linspace(0.35, 0.8, 1000)
        reverse_betas_str = ""
        keep_context = False
        folder_path = ("results/infN-correlations_from_weights-" + str(correlations_from_weights)
                       + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                       + str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                       + "/num_feat_patterns-" + str(num_feat_patterns) + "-normalize_weights-"
                       + normalize_weights_str + "-reorder_weights-" + str(int(reorder_weights))
                       + "-num_segments_corrs-" + str(num_segments_corrs) + "-pe_mode-" + str(pe_mode)
                       + gaussian_scale_name_str + "/max_sim_steps-" + str(max_sim_steps)
                       + save_non_transient_str + "-context_size-" + str(context_size)
                       + "/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1])
                       + "-num_betas-" + str(len(beta_list)) + f"{reverse_betas_str}-keep_context-"
                       + str(int(keep_context)))


        print(folder_path)

        # title = (f"MODE={correlations_from_weights} CONTEXT={context_size} NUM_PATTERNS={num_feat_patterns} SEED={seed} "
        #          f"BETA={beta} NUM_TRANSIENT={num_transient_steps_traj}")
        title = fr"$\beta={beta}$"


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


            plot_save_statistics(HT.mf_statistics[stat_name][num_transient_steps_traj:,:], stat_name,
                                 num_feat_patterns, max_sim_steps-num_transient_steps-num_transient_steps_traj,
                                 min_num_step=num_transient_steps+num_transient_steps_traj, title=title,
                                 save_path=plot_save_path_traj, save_not_plot=save_not_plot)

            plot_save_fft(HT.mf_statistics[stat_name][num_transient_steps_traj:,:], stat_name, num_feat_patterns,
                          max_sim_steps - num_transient_steps - num_transient_steps_traj,
                          show_max_num_patterns=num_feat_patterns, save_path=plot_save_path_fft,
                          save_not_plot=save_not_plot)

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
                        tag_names=stats_to_plot,  beta=beta, save_path=plot_save_path_plane, save_not_plot=save_not_plot)
