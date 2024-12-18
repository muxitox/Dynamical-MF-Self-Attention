import copy

import numpy as np
from models.HopfieldTransformerPE import HopfieldTransformer
from models.HopfieldTransformerMFInfNPE_old import HopfieldTransformerMFInfNPE

from bifurcation_diagrams import define_ini_token
from models.Embedding import Embedding
from plotting.plotting import plot_2_statistics, plot_save_statistics, plot_save_plane
import time
from utils import create_dir

if __name__ == "__main__":

    ######
    # General variables
    ########
    seed = 1
    beta_att = 2.2
    beta_o = 1.26405
    scaling_att = 100
    positional_embedding_size = 2
    scaling_o = 1
    normalize_weights_str_att = "N**2*np.sqrt(M)"
    normalize_weights_str_o = "N"
    min_saved_step = 100000
    saved_steps = 20000
    max_sim_steps = min_saved_step + saved_steps
    num_feat_patterns = 3
    context_size = 2 ** positional_embedding_size


    # Define list of possible initial tokens
    np.random.seed(0)
    num_ini_tokens = 1
    ini_token_idx = 0
    # Define wether to use a random initial token or to choose it from any of the patterns (Wo, Wv, Wq, Wk)
    ini_token_from_w = 1  # If 1, it will use the ini_token_idx pattern of W^o

    #################
    # MF Transformer
    #################

    correlations_from_weights = 3
    tentative_semantic_embedding_size = 99
    epsilon_pe = 0.02
    pe_mode = 0
    compute_inf_normalization = True

    # Instantiate vocabulary
    vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size)
    # We don't initialize the vocab as it's more efficient to work without a dict with the MF implementation

    # Create seed for reproducibility
    np.random.seed(seed)

    # Create Hopfield Transformer Class
    HTMF = HopfieldTransformerMFInfNPE(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                     positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                     context_size=context_size, max_sim_steps=max_sim_steps,
                                     min_saved_step=min_saved_step,
                                     normalize_weights_str_att=normalize_weights_str_att,
                                     normalize_weights_str_o=normalize_weights_str_o,
                                     correlations_from_weights=correlations_from_weights,
                                     semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                     epsilon_pe=epsilon_pe, pe_mode=pe_mode,
                                     compute_inf_normalization=compute_inf_normalization,
                                     scaling_o=scaling_o,
                                     scaling_att=scaling_att)

    # Define ini token
    ini_tokens_list = np.random.randint(2, size=(
        num_ini_tokens, tentative_semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    x0 = define_ini_token(ini_token_from_w, HTMF, ini_token_idx, ini_tokens_list)
    HTMF.reset_data()


    print("Simulating MF Transformer...")

    start = time.time()
    # HTMF.simulate(x0, max_steps=max_sim_steps)
    end = time.time()
    print("Done...")
    print("MF Time in [s]", end - start)


    #################
    # STD Transformer
    #################

    semantic_embedding_size = tentative_semantic_embedding_size * 10000

    # Instantiate vocabulary
    embedding_size = semantic_embedding_size + positional_embedding_size
    vocab = Embedding(semantic_embedding_size, positional_embedding_size)
    # We don't initialize the vocab as it's more efficient to work without a dict with the MF implementation


    reorder_weights = False

    # Create seed for reproducibility
    np.random.seed(seed)

    HT = HopfieldTransformer(beta_o, beta_att, num_feat_patterns, embedding_size, vocab, context_size,
                             max_sim_steps=max_sim_steps, min_saved_step=min_saved_step,
                             normalize_weights_str_att=normalize_weights_str_att,
                             normalize_weights_str_o=normalize_weights_str_o, reorder_weights=False, pe_mode=0,
                             epsilon_pe=epsilon_pe, weights_from_segments=True, scaling_o=scaling_o,
                             scaling_att=scaling_att, num_segments_corrs=3, model_to_replicate_corrs=HTMF)

    x0 = define_ini_token(ini_token_from_w, HT, ini_token_idx, ini_tokens_list)


    results_std_folder = f"results_std/beta{beta_o}/"
    create_dir(results_std_folder)


    num_std_runs = 1

    # Create variables for saving the mean statistics over experiments
    mean_mf_statistics = {}
    statistics_names = ["mo", "mo_se", "mv", "mq", "mk", "att"]
    for name_i in statistics_names:
        mean_mf_statistics[name_i] = np.zeros((saved_steps, num_feat_patterns))

    for run_i in range(num_std_runs):
        print(f"Simulating Standard Transformer {run_i + 1}...")
        start = time.time()
        HT.reset_data()
        HT.simulate(x0, max_steps=max_sim_steps)
        end = time.time()

        for name_i in statistics_names:
            mean_mf_statistics[name_i] += HT.mf_statistics[name_i]
        print("Done...")
        print("Standard Time in [s]", end - start)

    for name_i in statistics_names:
        mean_mf_statistics[name_i] /= num_std_runs


    print("Check weight correlations")
    print(HT.Wo.shape)
    print(HT.even_corr_o_o, HT.even_corr_o_o == HTMF.even_corr_o_o, 'mo')
    print(HT.even_corr_o_v, HT.even_corr_o_v == HTMF.even_corr_o_v, 'mv')
    print(HT.even_corr_o_q, HT.even_corr_o_q == HTMF.even_corr_o_q, 'mq')
    print(HT.even_corr_o_k, HT.even_corr_o_k == HTMF.even_corr_o_k, 'mk')

    if saved_steps < 50:
        print("MF Statistics mo_se")
        print(HTMF.mf_statistics["mo_se"])
        print()

        print("Standard Statistics mo_se")
        print(HT.mf_statistics["mo_se"])
        print()
        print()
        print()

        print("MF Statistics mv")
        print(HTMF.mf_statistics["mv"])
        print()

        print("Standard Statistics mv")
        print(HT.mf_statistics["mv"])
        print()
        print()
        print()

        print("MF Statistics mq")
        print(HTMF.mf_statistics["mq"])
        print()

        print("Standard Statistics mq")
        print(HT.mf_statistics["mq"])
        print()
        print()
        print()

        print("MF Statistics mk")
        print(HTMF.mf_statistics["mk"])
        print()

        print("Standard Statistics mk")
        print(HT.mf_statistics["mk"])
        print()
        print()
        print()


        print("MF Statistics att")
        print(HTMF.mf_statistics["att"])
        print()

        print("Standard Statistics att")
        print(HT.mf_statistics["att"] * HT.total_normalization_o)
        print()
        print()
        print()

    ##############
    # Plotting
    ##############

    print("Plotting statistics...")
    num_plotting_steps = saved_steps
    label_tag = ["MF", "STD"]
    beta_str = r" $\beta$ =" + str(beta_o)

    stats_to_show = ["mo_se", "mv", "mq", "mk", "att"]
    for stat_name in stats_to_show:
        plot_window = 350
        offset = 0  # Offset the trajectory to visit different points

        # Define the steps to show
        plot_range = [offset, offset + plot_window]  # Define the steps to plot

        if plot_range[1] > saved_steps:
            raise Exception("The rightmost index is greater than the number of steps.")

        rg = range(plot_range[0], plot_range[1])

        plot_2_statistics(HTMF.mf_statistics[stat_name][rg, :], mean_mf_statistics[stat_name][rg, :],
                          stat_name, num_feat_patterns,
                          plot_window, label_tag, additional_msg=beta_str + " avg")

        plot_2_statistics(HTMF.mf_statistics[stat_name][rg, :],
                          HT.mf_statistics[stat_name][rg, :], stat_name, num_feat_patterns,
                          plot_window, label_tag, additional_msg=beta_str + " 1traj")

        show_1_feat = 0  # Defines that it's only going to show 1 feature and what's its index

        # # Plot the MF trajectory
        # plot_save_statistics(HTMF.mf_statistics[stat_name], stat_name, num_feat_patterns,
        #                      saved_steps, min_num_step=0,
        #                      show_max_num_patterns=num_feat_patterns,
        #                      save_not_plot=False, title=None,
        #                      plot_hilbert=False, show_1_feat=show_1_feat)

        # plot_save_statistics(mean_mf_statistics[stat_name], stat_name, num_feat_patterns,
        #                      saved_steps, min_num_step=0,
        #                      show_max_num_patterns=num_feat_patterns,
        #                      save_not_plot=False, title=None,
        #                      plot_hilbert=False, show_1_feat=show_1_feat)


    # Plot planes

    # MF
    feat_idx = [[0], [1]]
    stats_to_plot = [["mo_se"], ["mo_se"]]
    stat_results_beta_list_0 = [HTMF.mf_statistics[stats_to_plot[feat_idx[0][0]][0]]]
    stat_results_beta_list_1 = [HTMF.mf_statistics[stats_to_plot[feat_idx[1][0]][0]]]
    # Plot plane
    plot_save_plane(stat_results_beta_list_0,
                    stat_results_beta_list_1, saved_steps, feat_idx,
                    tag_names=stats_to_plot, save_path=None, save_not_plot=False,
                    title=rf"$\beta={beta_o}$ MF", larger_dots=False)

    # Isolated traj 1
    feat_idx = [[0], [1]]
    stats_to_plot = [["mo_se"], ["mo_se"]]
    stat_results_beta_list_0 = [HT.mf_statistics[stats_to_plot[feat_idx[0][0]][0]]]
    stat_results_beta_list_1 = [HT.mf_statistics[stats_to_plot[feat_idx[1][0]][0]]]
    # Plot plane
    save_not_plot = True
    plane_savepath = results_std_folder + f"run_{run_i}_plane.pdf"
    title = fr"$\beta$ = {round(beta_o, 5)}"
    plot_save_plane(stat_results_beta_list_0,
                    stat_results_beta_list_1, saved_steps, feat_idx,
                    tag_names=stats_to_plot, save_path=plane_savepath, save_not_plot=save_not_plot,
                    title=title, larger_dots=False)

    # Isolated traj 1
    feat_idx = [[0], [1]]
    stats_to_plot = [["mo_se"], ["mo_se"]]
    stat_results_beta_list_0 = [mean_mf_statistics[stats_to_plot[feat_idx[0][0]][0]]]
    stat_results_beta_list_1 = [mean_mf_statistics[stats_to_plot[feat_idx[1][0]][0]]]
    # Plot plane


    plot_save_plane(stat_results_beta_list_0,
                    stat_results_beta_list_1, saved_steps, feat_idx,
                    tag_names=stats_to_plot, save_path=None, save_not_plot=False,
                    title=rf"$\beta={beta_o}$ STD (avg trajs)", larger_dots=False)


    print("Done.")
