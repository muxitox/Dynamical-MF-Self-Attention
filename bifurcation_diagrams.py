import numpy as np
from models.Embedding import Embedding
from models.HopfieldSelfAttentionNNMFInfNPE import HopfieldSelfAttentionNNMFInfNPE
from models.HopfieldSelfAttentionNNMFPE import HopfieldSelfAttentionNNMFPE
from plotting.plotting import (plot_save_plane, plot_lyapunov_graphs, plot_bifurcation_lyapunov, plot_bifurcation_diagram,
                               filter_bifurcation_diagram_by_couting)
import os
import time
import copy
from utils import create_dir, create_dir_from_filepath, load_context
import matplotlib.pyplot as plt
import yaml
import datetime


def create_pathname_inf_betas(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                              cfg, seed, ini_token_idx):
    """
    Given the experiment parameters, creates a path to save it.
    The code is a bit intrincate for back-compatibility with older experiments.
    """

    if cfg["bifurcation_mode"] == "betas":
        results_folder = "results_parallel_v3"
        beta_string = ("/min_beta-" + str(worker_values_list[0]) + "-max_beta-" + str(worker_values_list[-1]) +
                       "-num_betas-" + str(len(worker_values_list)))
    elif cfg["bifurcation_mode"] == "out":
        results_folder = "results_out_parallel_v3"
        beta_string = (
                    "/beta_att-" + str(cfg["beta_att"]) + "-min_beta_o-" + str(worker_values_list[0]) + "-max_beta_o-" +
                    str(worker_values_list[-1]) + "-num_betas-" + str(len(worker_values_list)))
    elif cfg["bifurcation_mode"] == "att":
        results_folder = "results_att_parallel_v3"
        beta_string = (
                    "/beta_o-" + str(cfg["beta_o"]) + "-min_beta_att-" + str(worker_values_list[0]) + "-max_beta_att-" +
                    str(worker_values_list[-1]) + "-num_betas-" + str(len(worker_values_list)))
    else:
        raise Exception("mode not recognized (not one of [\"betas\", \"out\", \"att\", \"pe\"])")

    gaussian_scale_str = cfg["gaussian_scale"]
    if cfg["correlations_from_weights"] != 0:
        gaussian_scale_name_str = ""
    else:
        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

    num_transient_steps = cfg["num_transient_steps"]
    if cfg["save_non_transient"] == True:
        save_non_transient_str = ""
    else:
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"

    if cfg["normalize_weights_str_o"] == cfg["normalize_weights_str_att"]:
        normalize_weights_name_str = "-normalize_weights-" + cfg["normalize_weights_str_att"]
    else:
        normalize_weights_name_str = ("-normalize_weights_att-" + cfg["normalize_weights_str_att"] +
                                      "-normalize_weights_o-" + cfg["normalize_weights_str_o"])

    scaling_str = ""
    if cfg["scaling_o"] != 1:
        scaling_str += "-scaling_o-" + str(cfg["scaling_o"])
    if cfg["scaling_att"] != 1:
        scaling_str += "-scaling_att-" + str(cfg["scaling_att"])

    compute_inf_normalization_str = ""
    if cfg["compute_inf_normalization"]:
        compute_inf_normalization_str = "-inf_norm"

    load_chpt_str = ""
    if cfg["load_chpt"]:
        load_chpt_str = "-load_from_chpt-"

    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = (f"{results_folder}/infN-correlations_from_weights-" + str(cfg["correlations_from_weights"])
                   + "-se_size-" + str(cfg["semantic_embedding_size"]) + "-pe_size-"
                   + str(positional_embedding_size) + "-se_per_contribution-" + str(1 - cfg["epsilon_pe"])
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(cfg["reorder_weights"])) + "-num_segments_corrs-" + str(cfg["num_segments_corrs"])
                   + "-pe_mode-" + str(cfg["pe_mode"]) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(cfg["max_sim_steps"]) + save_non_transient_str + "-context_size-" + str(context_size)
                   + beta_string + load_chpt_str)

    # Set up some more variables for saving purposes
    ini_token_mode_str = ""
    if not cfg["load_chpt"]:
        ini_token_from_w = cfg["ini_token_from_w"]
        if cfg["ini_token_from_w"] != 0:
            ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
    folder_path = (folder_path + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx)
                   + ini_token_mode_str + "/")

    return folder_path


def create_pathname_inf_pes(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                            cfg, seed, ini_token_idx):
    """
    Given the experiment parameters, creates a path to save it.
    The code is a bit intrincate for back-compatibility with older experiments.
    """

    epsilon_pe_string = ("/min_epsilon_pe-" + str(worker_values_list[0]) + "-min_epsilon_pe-" +
                         str(worker_values_list[-1]) + "-num_pes-" + str(len(worker_values_list)))

    gaussian_scale_str = cfg["gaussian_scale_str"]
    if cfg["correlations_from_weights"] != 0:
        gaussian_scale_name_str = ""
    else:
        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

    num_transient_steps = cfg["num_transient_steps"]
    if cfg["save_non_transient"]:
        save_non_transient_str = ""
    else:
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"

    if cfg["normalize_weights_str_o"] == cfg["normalize_weights_str_att"]:
        normalize_weights_name_str = "-normalize_weights-" + cfg["normalize_weights_str_att"]
    else:
        normalize_weights_name_str = ("-normalize_weights_att-" + cfg["normalize_weights_str_att"] +
                                      "-normalize_weights_o-" + cfg["normalize_weights_str_o"])

    scaling_str = ""
    if cfg["scaling_o"] != 1:
        scaling_str += "-scaling_o-" + str(cfg["scaling_o"])
    if cfg["scaling_att"] != 1:
        scaling_str += "-scaling_att-" + str(cfg["scaling_att"])

    compute_inf_normalization_str = ""
    if cfg["compute_inf_normalization"]:
        compute_inf_normalization_str = "-inf_norm"

    load_chpt_str = ""
    if cfg["load_chpt"]:
        load_chpt_str = "-load_from_chpt-"

    if cfg["beta_att"] == cfg["beta_o"]:
        beta_string = "-beta-" + str(cfg["beta_att"])
    else:
        beta_string = "-beta_o-" + str(cfg["beta_o"]) + "-beta_att-" + str(cfg["beta_att"])

    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = ("results_pe_parallel_v3/infN-correlations_from_weights-" + str(cfg["correlations_from_weights"])
                   + "-se_size-" + str(cfg["semantic_embedding_size"]) + "-pe_size-"
                   + str(positional_embedding_size) + beta_string
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(cfg["reorder_weights"])) + "-num_segments_corrs-" + str(cfg["num_segments_corrs"])
                   + "-pe_mode-" + str(cfg["pe_mode"]) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(cfg["max_sim_steps"]) + save_non_transient_str + "-context_size-" + str(context_size)
                   + epsilon_pe_string + load_chpt_str)

    # Set up some more variables for saving purposes
    ini_token_mode_str = ""
    if not cfg["load_chpt"]:
        ini_token_from_w = cfg["ini_token_from_w"]
        if cfg["ini_token_from_w"] != 0:
            ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
    folder_path = (folder_path + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx)
                       + ini_token_mode_str + "/")

    return folder_path


def create_pathname(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                    cfg, seed, ini_token_idx):
    if cfg["bifurcation_mode"] == "pe":
        pathname = create_pathname_inf_pes(num_feat_patterns, positional_embedding_size, context_size,
                                           worker_values_list, cfg, seed, ini_token_idx)
    else:
        pathname = create_pathname_inf_betas(num_feat_patterns, positional_embedding_size, context_size,
                                             worker_values_list, cfg, seed, ini_token_idx)

    return pathname


def define_ini_token(ini_token_from_w, HT, ini_token_idx, ini_tokens_list):
    """
    Defines how to set the initial token
    """
    if ini_token_from_w == 0:
        # Encode initial token with position 0
        x0 = copy.deepcopy(ini_tokens_list[ini_token_idx])
    elif ini_token_from_w == 1:
        x0 = copy.deepcopy(HT.Wo[ini_token_idx])
    elif ini_token_from_w == 2:
        x0 = copy.deepcopy(HT.Wv[ini_token_idx])
    elif ini_token_from_w == 3:
        x0 = copy.deepcopy(HT.Wq[ini_token_idx])
    elif ini_token_from_w == 4:
        x0 = copy.deepcopy(HT.Wk[ini_token_idx])
    else:
        raise Exception("ini_token_idx is not in the range [0,4]")

    return x0

def initialize_bifurcation_variable(HT, worker_values_list, worker_id, mode):
    if mode == "betas":
        HT.set_betas(worker_values_list[worker_id], worker_values_list[worker_id])
    elif mode == "out":
        HT.set_beta_o(worker_values_list[worker_id])
    elif mode == "att":
        HT.set_beta_att(worker_values_list[worker_id])
    elif mode == "pe":
        HT.set_epsilon_pe(worker_values_list[worker_id])
    else:
        raise Exception("mode not recognized (not one of [\"betas\", \"out\", \"att\", \"pe\"])")


def plot_lowres_planes(worker_values_list, beta_idx, cfg, folder_path, image_format = ".jpeg"):
    # For internal use mostly, to decide the final plots. Creates low resolution images of the planes.
    stats_data_path = (folder_path + "/stats" + "/beta_idx-" + str(beta_idx)
                       + ".npz")

    # Load data
    data = np.load(stats_data_path)

    plot_save_path_plane = (folder_path + f"/indiv_lowres_traj/planes/"
                            + f"/plane-beta-{worker_values_list[beta_idx]}" + "-transient_steps-" +
                            str(cfg["num_transient_steps"]) + image_format)

    create_dir_from_filepath(plot_save_path_plane)

    nrows = 2
    ncols = 3
    dpi = 17
    fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8), constrained_layout=True, dpi=dpi)

    # Define the statistics you want to plot against each other
    # In this case the feature mo with only the semantic information
    stats_to_plot = [["mo_se", "mo_se"], ["mo_se", "mo_se"], ["mo_se", "mo_se"],
                     ["att", "att"], ["att", "att"], ["att", "att"]]
    # Define the index of the features you want to compare against each other
    feat_idx = [[0, 1], [0, 2], [1, 2], [0, 1], [0, 2], [1, 2]]

    flat_ax = ax.ravel()

    for plot_i in range(len(flat_ax)):
        # Load needed statistics
        stat_results_beta_0 = data[stats_to_plot[plot_i][0]][:, feat_idx[plot_i][0]]
        stat_results_beta_1 = data[stats_to_plot[plot_i][0]][:, feat_idx[plot_i][1]]

        plot_save_plane(stat_results_beta_0,
                        stat_results_beta_1, cfg["max_sim_steps"] - cfg["num_transient_steps"], feat_idx[plot_i],
                        flat_ax[plot_i], tag_names=stats_to_plot[plot_i])

        fig.savefig(plot_save_path_plane, bbox_inches='tight')


def plot_lowres_lyapunov(S_i_sum, worker_values_list, beta_idx, cfg,
                         folder_path, image_format=".jpeg"):


    plot_save_path_lya = (folder_path + f"/indiv_lowres_traj/lyapunov/"
                            + f"/lyapunovtrace-beta-{worker_values_list[beta_idx]}" + "-transient_steps-" +
                            str(cfg["num_transient_steps"]) + image_format)

    create_dir_from_filepath(plot_save_path_lya)

    # Plot lyapunov related statistics
    plot_lyapunov_graphs(S_i_sum, cfg, worker_values_list[beta_idx],
                         save_not_plot=True, save_path=plot_save_path_lya, lowres=True)

def compute_max_min(x_list, folder_path, min_bidx, feat_name):
    """
    Computes the max and min values for all the betas
    """
    max_y = - np.inf
    min_y = np.inf
    for idx in range(len(x_list)):
        b_idx = min_bidx + idx
        stats_data_path = (folder_path + "/stats" + "/beta_idx-" + str(b_idx)
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

def filter_y_values_by_0_plane(results_y_list, feat, filter_idx, filtering_range):
    filtering_values = results_y_list[:, filter_idx]
    zero_intersect = np.where(np.logical_and(filtering_values >= -filtering_range,
                                             filtering_values <= filtering_range))[0]
    return results_y_list[zero_intersect, feat]

def get_0_plane_filter_intersection(idx, folder_path, min_bidx, feat_name,
                        num_transient_steps, feat, y_resolution, filter_idx, filtering_range, max_y):
    """
    Function to compute the intersection with the 0 plane for each x value.
    """

    b_idx = min_bidx + idx
    stats_data_path = (folder_path + "/stats" + "/beta_idx-" + str(b_idx)
                       + ".npz")

    # Load data
    data = np.load(stats_data_path)
    results_y_list = data[f"{feat_name}_results_beta"]

    values_feat_filtered = filter_y_values_by_0_plane(results_y_list[num_transient_steps:], feat,
                                                      filter_idx, filtering_range)

    values_feat_filtered_quantized = (np.unique((y_resolution * (values_feat_filtered / max_y + 1) / 2).astype(int))
                                      / y_resolution * 2 - 1) * max_y

    values_feat = results_y_list[num_transient_steps:, feat]
    values_feat_quantized = (np.unique((y_resolution * (values_feat / max_y + 1) / 2).astype(int))
                             / y_resolution * 2 - 1) * max_y
    unique_len = len(values_feat_quantized)

    return values_feat_filtered_quantized, values_feat_quantized, unique_len


def filter_bifurcation_diagram(beta_list_to_plot, beta_idx_to_filter, min_beta_idx, exp_dir, stat_name,
                               num_transient_steps_plot_arg, feat_to_plot, filter_by_feat, filtering_range, max_y,
                               local_ax):

    y_resolution_filtered_plot = 5001
    # Filter y values of `feat_to_plot` py 0 plane of `filter_by_feat`
    values_feat_filtered_quantized, values_feat_quantized, unique_len = \
        (get_0_plane_filter_intersection(beta_idx_to_filter, exp_dir, min_beta_idx,
                                         stat_name, num_transient_steps_plot_arg, feat_to_plot,
                                         y_resolution_filtered_plot, filter_by_feat,
                                         filtering_range, max_y))

    # Old heuristic: if the number of different quantized points is < 80, assume it's periodic
    filter_periodic = 80
    filter_bifurcation_diagram_by_couting(beta_list_to_plot, beta_idx_to_filter,
                                          values_feat_filtered_quantized, values_feat_quantized,
                                          unique_len, local_ax, filter_periodic=filter_periodic)


def runner(worker_values_list, worker_id, cfg, exp_dir, stats_to_save_plot):
    """

    :return:
    """

    if worker_id == 0:
        # If you are node 0, save the config
        file = open(f"{exp_dir}/cfg.yaml", "w")
        yaml.dump(cfg, file)
        file.close()


    vocab = Embedding(cfg["semantic_embedding_size"], cfg["positional_embedding_size"])

    # Seed equal to 0 for initial token set up
    np.random.seed(0)
    num_ini_tokens = 10  # Number of candidate initial tokens

    ini_tokens_list = np.random.randint(2, size=(
        num_ini_tokens, cfg["semantic_embedding_size"] + cfg["positional_embedding_size"])) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -cfg["positional_embedding_size"]:] = -1

    min_saved_step = 0
    if not cfg["save_non_transient"]:
        min_saved_step = cfg["num_transient_steps"]

    # Create path for stats saving
    folder_path_stats = exp_dir + "/stats/"
    create_dir(folder_path_stats)

    compute_lyapunov = cfg["compute_lyapunov"]

    # Define the seed that will create the weights/correlations
    np.random.seed(cfg["seed"])

    if cfg["inf_mode"]:
        # Initialize the Hopfield Transformer class. \beta will be set afterwards
        HT = HopfieldSelfAttentionNNMFInfNPE(cfg["beta_o"], cfg["beta_att"], num_feat_patterns=cfg["num_feat_patterns"],
                                             positional_embedding_bitsize=cfg["positional_embedding_size"], vocab=vocab,
                                             context_size=cfg["context_size"], max_sim_steps=cfg["max_sim_steps"],
                                             min_saved_step=min_saved_step,
                                             normalize_weights_str_att=cfg["normalize_weights_str_att"],
                                             normalize_weights_str_o=cfg["normalize_weights_str_o"],
                                             reorder_weights=cfg["reorder_weights"],
                                             correlations_from_weights=cfg["correlations_from_weights"],
                                             num_segments_corrs=cfg["num_segments_corrs"], pe_mode=cfg["pe_mode"],
                                             semantic_embedding_bitsize=cfg["semantic_embedding_size"],
                                             epsilon_pe=cfg["epsilon_pe"],
                                             gaussian_scale_str=cfg["gaussian_scale"],
                                             compute_inf_normalization=cfg["compute_inf_normalization"],
                                             N_normalization=9999,
                                             scaling_o=cfg["scaling_o"],
                                             scaling_att=cfg["scaling_att"])
    else:
        HT = HopfieldSelfAttentionNNMFPE(cfg["beta_o"], cfg["beta_att"], num_feat_patterns=cfg["num_feat_patterns"],
                                         embedding_size=cfg["semantic_embedding_size"] + cfg["positional_embedding_size"],
                                         vocab=vocab, context_size=cfg["context_size"], max_sim_steps=cfg["max_sim_steps"],
                                         min_saved_step=min_saved_step,
                                         normalize_weights_str_att=cfg["normalize_weights_str_att"],
                                         normalize_weights_str_o=cfg["normalize_weights_str_o"],
                                         reorder_weights=cfg["reorder_weights"],
                                         scaling_o=cfg["scaling_o"],
                                         scaling_att=cfg["scaling_att"],
                                         weights_from_segments=cfg["weights_from_segments"])

    # Initialize structure for saving the results for each beta
    # Fields are left with empty list if not requested
    results_beta = {}
    for stat_name in HT.statistics_names:
        results_beta[stat_name] = []
    results_beta["S"] = []
    results_beta["S_inf_flag"] = []


    # Set either both betas, one of them or epsilon from the positional encoding
    initialize_bifurcation_variable(HT, worker_values_list, worker_id, cfg["bifurcation_mode"])

    print(f"Computing seed ", cfg["seed"] ,f"beta {worker_id + 1}/{len(worker_values_list)}", flush=True)

    # Reset data structures
    HT.reset_data()

    # Measure only simulation time
    start = time.time()

    if cfg["load_chpt"]:
        # Load checkpoint from last beta
        att_window, pe_window = load_context(cfg["chpt_path"])
        # Simulate from context
        HT.simulate(att_window, pe_window, max_steps=cfg["max_sim_steps"], compute_lyapunov=cfg["compute_lyapunov"])
    else:
        # Define the initial token. x0 is only used if load_from_context_mode!=2
        x0 = define_ini_token(cfg["ini_token_from_w"], HT, cfg["ini_token_idx"], ini_tokens_list)
        if cfg["ini_token_from_w"] != 0:  # Otherwise it's already set
            x0[-cfg["positional_embedding_size"]:] = -1  # Initialize position to -1

        # Simulate for max_sim_steps steps from x0
        HT.simulate_from_token(x0, max_steps=cfg["max_sim_steps"], compute_lyapunov=cfg["compute_lyapunov"])

    end = time.time()
    elapsed_time = end - start
    print("Simulation: elapsed time in minutes", elapsed_time / 60)
    print("Simulation: elapsed time in hours", elapsed_time / 3600)


    for stat_name in stats_to_save_plot:
        # Accumulate results in a var of beta_list length
        results_beta[stat_name] = np.copy(HT.mf_statistics[stat_name])

    stats_data_path = (folder_path_stats + "beta_idx-" + str(worker_id) + ".npz")

    if compute_lyapunov:
        results_beta["S"] = HT.S
        results_beta["S_inf_flag"] = HT.S_inf_flag


    # Save results
    print("Saving results in ", os.path.abspath(stats_data_path))
    np.savez_compressed(stats_data_path,
                        mo_results_beta=results_beta["mo"],
                        mo_se_results_beta=results_beta["mo_se"],
                        mv_results_beta=results_beta["mv"],
                        mq_results_beta=results_beta["mq"],
                        mk_results_beta=results_beta["mk"],
                        att_results_beta=results_beta["att"],
                        S=results_beta["S"],
                        S_inf_flag=results_beta["S_inf_flag"],
                        simulation_time=elapsed_time
                        )

    plot_lowres= True
    if plot_lowres:
        plot_lowres_planes(worker_values_list, worker_id, cfg, exp_dir)

        if cfg["compute_lyapunov"]:
            plot_lowres_lyapunov(HT.S_i_sum, worker_values_list, worker_id, cfg, exp_dir)

    print(f"Saved stats num_feat_patterns", cfg["num_feat_patterns"],  "seed ", cfg["seed"])


def plotter(worker_values_list, cfg, exp_dir,
            stats_to_save_plot, min_max_beta_to_show=None, show_title=False):
    # Set up some parameters for loading the experiments statistics
    if min_max_beta_to_show is None:
        min_beta_idx = 0
        max_beta_idx = None
    else:  # In this else, if set, we can zoom_in the bif. diagram but without much resolution
        min_beta_idx = np.searchsorted(worker_values_list, min_max_beta_to_show[0])
        max_beta_idx = np.searchsorted(worker_values_list, min_max_beta_to_show[1]) + 1

    if cfg["save_non_transient"] == True:
        num_transient_steps_plot_arg = cfg["num_transient_steps"]
    else:
        num_transient_steps_plot_arg = 0

    save_not_plot = cfg["save_not_plot"] # If true -> saves, if false -> plots
    save_not_plot = False

    # image_format = ".jpeg"
    image_format = ".pdf"

    correlations_from_weights = cfg["correlations_from_weights"]
    filtering_range = cfg["filtering_range"]

    # Get the requested list of betas
    beta_list_to_plot = worker_values_list[min_beta_idx:max_beta_idx]

    # Hardcoded, :(, select what you want to plot
    operations = ["bifurcation", "lyapunov"]
    # operations = ["lyapunov"]

    ###############################
    # Plot the bifurcation diagrams
    ###############################

    # The value of the list is the index of the feature to plot.
    feat_to_plot_list = [0, 0, 1, 2]
    filter_by_feat_list = [1, 2, 0, 0]

    if "bifurcation" in operations:
        for stat_name in stats_to_save_plot:

            # Create folder if it does not exist and we are saving the image
            if save_not_plot and (not os.path.exists(exp_dir + f"/{stat_name}/")):
                os.makedirs(exp_dir + f"/{stat_name}/")

            for plot_i in range(len(feat_to_plot_list)):
                # The feature for which we are computing the bifurcation diagram
                feat_to_plot = feat_to_plot_list[plot_i]
                # The feature we use for intersection with the 0 plane
                filter_by_feat = filter_by_feat_list[plot_i]

                # Title for internal use
                if show_title:
                    title = (
                        "CORRm=" + str(cfg["correlations_from_weights"]) + " CTX=" + str(cfg["context_size"])
                        + " NUM_PAT=" + str(cfg["num_feat_patterns"]) + "SEED=" + str(cfg["seed"]) +
                        f" Filter={filtering_range}")
                else:
                    title = None

                # Save path
                filtered_plot_save_path = (exp_dir + f"/{stat_name}/" +
                                           "transient_steps-" + str(cfg["num_transient_steps"]) +
                                           "-filter_idx-" + str(filter_by_feat) +
                                           "-filter_rg-" + str(filtering_range) + image_format)

                # Plotting and saving
                print("Creating and saving/plotting diagram")

                col_size = 5
                row_size = 4
                dpi = 250
                # Using subplot to generalize the behavior if we want to compute a more complex plot
                fig, ax = plt.subplots(1, 1, figsize=(col_size, row_size), constrained_layout=True, dpi=dpi)

                # Compute min max range to define the im_array properly
                min_y, max_y = compute_max_min(beta_list_to_plot, exp_dir, min_beta_idx, stat_name)

                # Plot basic bifurcation diagram
                plot_bifurcation_diagram(feat_to_plot, beta_list_to_plot, num_transient_steps_plot_arg, stat_name,
                                         exp_dir, ax, min_y, max_y, x_label=r'$\beta$', min_bidx=min_beta_idx)

                if not save_not_plot:
                    print("Plotting bifurcation diagram before filtering")
                    plt.show()

                # Filter the bifurcation diagram using counting mechanisms
                for beta_idx_to_filter in range(len(beta_list_to_plot)):
                    if beta_idx_to_filter % 100 == 0:
                        print(f"Filtering {beta_idx_to_filter + 1}/{len(beta_list_to_plot)}")
                    filter_bifurcation_diagram(beta_list_to_plot, beta_idx_to_filter, min_beta_idx, exp_dir, stat_name,
                               num_transient_steps_plot_arg, feat_to_plot, filter_by_feat, filtering_range, max_y,
                               ax)

                if title is not None:
                    fig.suptitle(title)

                if save_not_plot:
                    fig.savefig(filtered_plot_save_path, bbox_inches='tight')
                else:
                    print("Plotting filtered bifurcation diagram")
                    plt.show()

    ####################################################
    # Plot the stats related with the Lyapunov exponents
    ####################################################

    if "lyapunov" in operations:
        lya_hist_save_basepath = (exp_dir + f"/Lyapunov/")

        if save_not_plot and (not os.path.exists(lya_hist_save_basepath)):
            os.makedirs(lya_hist_save_basepath)

        print("Creating and saving/plotting lyapunov statistics")
        plot_bifurcation_lyapunov(beta_list_to_plot, cfg["num_feat_patterns"], cfg["context_size"], exp_dir, lya_hist_save_basepath,
                      save_not_plot=save_not_plot, title=None, min_bidx=min_beta_idx)



if __name__ == "__main__":

    # Load cfg
    cfg_path = 'cfgs/bif_diagram_inf_0.yaml'
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Create folder to save the results
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    exp_dir = f"results_parallel_v3/{date_str}/"
    print("Creating dir for saving the experiments in", exp_dir)
    create_dir(exp_dir)

    # Create the variables from the experiment that are not set up in the yaml cfg.
    cfg["num_bifurcation_values"] = 10  # Number of x values to examine in the bifurcation diagram

    worker_values_list = np.linspace(cfg["min_bifurcation_value"], cfg["max_bifurcation_value"],
                                     cfg["num_bifurcation_values"])  # Betas or Epsilon values

    # Add remaining config values to cfg
    cfg["positional_embedding_size"] = 2
    cfg["context_size"] = 2 ** cfg["positional_embedding_size"]

    cfg["seed"] = 1  # List of seeds to review
    cfg["num_feat_patterns"] = 3  # List of number of features for which to initialize the model
    cfg["ini_token_idx"] = 0

    show_title = False  # Whether to plot a title with the characteristics of the experiment. For internal use mostly.

    if cfg["context_size"] > 2 ** cfg["positional_embedding_size"]:
        raise ("The positional embedding cannot cover the whole context size.")
    if cfg["num_transient_steps"] > cfg["max_sim_steps"]:
        raise ("You cannot discard more timesteps than you are simulating.")

    stats_to_save_plot = ["mo_se", "att"]

    start = time.time()

    # Then compute the rest of the betas, setting the initial context to the last beta one
    for worker_id in range(cfg["num_bifurcation_values"]):
        runner(worker_values_list, worker_id, cfg, exp_dir, stats_to_save_plot)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    # Once computed, load checkpoints and plot them
    plotter(worker_values_list, cfg, exp_dir, stats_to_save_plot, show_title=show_title)
