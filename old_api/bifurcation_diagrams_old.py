import numpy as np
from models.Embedding import Embedding
from models.HopfieldTransformerMFInfNPE import HopfieldTransformerMFInfNPE
from models.HopfieldTransformerMFPE import HopfieldTransformerMFPE
from plotting.plotting import plot_filtered_bifurcation_diagram_par_imshow
import os
import time
import copy
from utils import create_dir, create_dir_from_filepath
from plotting.plotting import plot_save_plane
import yaml


def create_pathname_inf_betas(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                              load_from_context_mode, cfg):
    """
    Given the experiment parameters, creates a path to save it.
    The code is a bit intrincate for back-compatibility with older experiments.
    """

    if cfg["bifurcation_mode"] == "betas":
        results_folder = "results_parallel"
        beta_string = ("/min_beta-" + str(worker_values_list[0]) + "-max_beta-" + str(worker_values_list[-1]) +
                       "-num_betas-" + str(len(worker_values_list)))
    elif cfg["bifurcation_mode"] == "out":
        results_folder = "results_out_parallel"
        beta_string = (
                    "/beta_att-" + str(cfg["beta_att"]) + "-min_beta_o-" + str(worker_values_list[0]) + "-max_beta_o-" +
                    str(worker_values_list[-1]) + "-num_betas-" + str(len(worker_values_list)))
    elif cfg["bifurcation_mode"] == "att":
        results_folder = "results_att_parallel"
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

    load_from_context_mode_str = ""
    if load_from_context_mode != 0:
        load_from_context_mode_str = "-load_from_context_mode-1"

    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = (f"{results_folder}/infN-correlations_from_weights-" + str(cfg["correlations_from_weights"])
                   + "-se_size-" + str(cfg["semantic_embedding_size"]) + "-pe_size-"
                   + str(positional_embedding_size) + "-se_per_contribution-" + str(1 - cfg["epsilon_pe"])
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(cfg["reorder_weights"])) + "-num_segments_corrs-" + str(cfg["num_segments_corrs"])
                   + "-pe_mode-" + str(cfg["pe_mode"]) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(cfg["max_sim_steps"]) + save_non_transient_str + "-context_size-" + str(context_size)
                   + beta_string + load_from_context_mode_str)

    return folder_path


def create_pathname_inf_pes(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                            load_from_context_mode, cfg):
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

    load_from_context_mode_str = ""
    if load_from_context_mode != 0:
        load_from_context_mode_str = "-load_from_context_mode-1"

    if cfg["beta_att"] == cfg["beta_o"]:
        beta_string = "-beta-" + str(cfg["beta_att"])
    else:
        beta_string = "-beta_o-" + str(cfg["beta_o"]) + "-beta_att-" + str(cfg["beta_att"])

    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = ("results_pe_parallel/infN-correlations_from_weights-" + str(cfg["correlations_from_weights"])
                   + "-se_size-" + str(cfg["semantic_embedding_size"]) + "-pe_size-"
                   + str(positional_embedding_size) + beta_string
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(cfg["reorder_weights"])) + "-num_segments_corrs-" + str(cfg["num_segments_corrs"])
                   + "-pe_mode-" + str(cfg["pe_mode"]) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(cfg["max_sim_steps"]) + save_non_transient_str + "-context_size-" + str(context_size)
                   + epsilon_pe_string + load_from_context_mode_str)

    return folder_path


def create_pathname(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                    load_from_context_mode, cfg):
    if cfg["bifurcation_mode"] == "pe":
        pathname = create_pathname_inf_pes(num_feat_patterns, positional_embedding_size, context_size,
                                           worker_values_list, load_from_context_mode, cfg)
    else:
        pathname = create_pathname_inf_betas(num_feat_patterns, positional_embedding_size, context_size,
                                             worker_values_list, load_from_context_mode, cfg)

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


def save_context(context_window, folder_path_chpt, beta_idx):
    """
    Saves the mean-field values associated to the context window
    """
    att_window, mo_window, mv_window, mq_window, mk_window, pe_window = context_window

    chpt_path = folder_path_chpt + f"/beta_idx-{beta_idx}_window_chpt.npz"

    np.savez_compressed(chpt_path,
                        att_window=att_window,
                        mo_window=mo_window,
                        mv_window=mv_window,
                        mq_window=mq_window,
                        mk_window=mk_window,
                        pe_window=pe_window)

def load_context(folder_path_chpt, beta_idx):
    """
    Load the mean-field values associated to the context window of a previous experiment.
    :param beta_idx index of the beta from which to load the context window
    """
    chpt_path = folder_path_chpt + f"/beta_idx-{beta_idx}_window_chpt.npz"

    cw = np.load(chpt_path)

    # We just need the attention and positional encodings

    return cw['att_window'], cw['pe_window']


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


def runner(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list,
           worker_id, cfg, stats_to_save_plot, load_from_context_mode=0):
    """

    :param load_from_context_mode: 0 -> don't load from context, 1 -> don't load from context but save your final context
                                   2-> load context from other experiment
    :return:
    """

    vocab = Embedding(cfg["semantic_embedding_size"], positional_embedding_size)

    # Seed equal to 0 for initial token set up
    np.random.seed(0)
    num_ini_tokens = 10  # Number of candidate initial tokens

    ini_tokens_list = np.random.randint(2, size=(
        num_ini_tokens, cfg["semantic_embedding_size"] + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1

    min_saved_step = 0
    if not cfg["save_non_transient"]:
        min_saved_step = cfg["num_transient_steps"]

    # Create root folder to later save and aggregate the results
    folder_path = create_pathname(num_feat_patterns, positional_embedding_size, context_size, worker_values_list, load_from_context_mode, cfg)

    folder_path_chpt = folder_path + "/chpt"
    folder_path = folder_path + "/stats"

    create_dir(folder_path)
    if load_from_context_mode == 1:
        create_dir(folder_path_chpt)

    # Define the seed that will create the weights/correlations
    np.random.seed(seed)

    if cfg["inf_mode"]:
        # Initialize the Hopfield Transformer class. \beta will be set afterwards
        HT = HopfieldTransformerMFInfNPE(cfg["beta_o"], cfg["beta_att"], num_feat_patterns=num_feat_patterns,
                                         positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                         context_size=context_size, max_sim_steps=cfg["max_sim_steps"],
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
        HT = HopfieldTransformerMFPE(cfg["beta_o"], cfg["beta_att"], num_feat_patterns=num_feat_patterns,
                                     embedding_size=cfg["semantic_embedding_size"] + positional_embedding_size,
                                     vocab=vocab, context_size=context_size, max_sim_steps=cfg["max_sim_steps"],
                                     min_saved_step=min_saved_step,
                                     normalize_weights_str_att=cfg["normalize_weights_str_att"],
                                     normalize_weights_str_o=cfg["normalize_weights_str_o"],
                                     reorder_weights=cfg["reorder_weights"],
                                     scaling_o=cfg["scaling_o"],
                                     scaling_att=cfg["scaling_att"],
                                     weights_from_segments=cfg["weights_from_segments"])

    # Initialize structure for saving the results for each beta
    results_beta = {}
    for stat_name in HT.statistics_names:
        results_beta[stat_name] = []

    # Set either both betas, one of them or epsilon from the positional encoding
    initialize_bifurcation_variable(HT, worker_values_list, worker_id, cfg["bifurcation_mode"])

    print(f"Computing seed {seed} beta {worker_id + 1}/{len(worker_values_list)}", flush=True)

    # Reset data structures
    HT.reset_data()

    # Define the initial token. x0 is only used if load_from_context_mode!=2
    x0 = define_ini_token(cfg["ini_token_from_w"], HT, ini_token_idx, ini_tokens_list)
    ini_token_from_w = cfg["ini_token_from_w"]
    if ini_token_from_w != 0:  # Otherwise it's already set
        x0[-positional_embedding_size:] = -1  # Initialize position to -1

    if load_from_context_mode == 0 or load_from_context_mode == 1:
        # Simulate for max_sim_steps steps from x0
        HT.simulate_from_token(x0, max_steps=cfg["max_sim_steps"])
        if load_from_context_mode == 1:
            # Save context reordered for a fresh start
            cw = HT.get_context_window()
            save_context(cw, folder_path_chpt, worker_id)
    elif load_from_context_mode == 2:
        # Load checkpoint from last beta
        att_window, pe_window = load_context(folder_path_chpt, len(worker_values_list) - 1)
        # Simulate from context
        HT.simulate(att_window, pe_window, max_steps=cfg["max_sim_steps"])

    for stat_name in stats_to_save_plot:
        # Accumulate results in a var of beta_list length
        results_beta[stat_name] = np.copy(HT.mf_statistics[stat_name])

    # Set up some more variables for saving purposes
    ini_token_mode_str = ""
    if ini_token_from_w != 0:
        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
    stats_data_path = (folder_path + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx)
                       + ini_token_mode_str + "-beta_idx-" + str(worker_id) + ".npz")

    # Save results
    print("Saving results in ", os.path.abspath(stats_data_path))
    np.savez_compressed(stats_data_path,
                        mo_results_beta=results_beta["mo"],
                        mo_se_results_beta=results_beta["mo_se"],
                        mv_results_beta=results_beta["mv"],
                        mq_results_beta=results_beta["mq"],
                        mk_results_beta=results_beta["mk"],
                        att_results_beta=results_beta["att"])

    print(f"Saved stats num_feat_patterns {num_feat_patterns}, seed {seed}, ini_token_idx {ini_token_idx}")


def plotter(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list, cfg,
            stats_to_save_plot, load_from_context_mode=0, min_max_beta_to_show=None, show_title=False):
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

    # image_format = ".jpeg"
    image_format = ".pdf"

    # Create pathname
    folder_path = create_pathname(num_feat_patterns, positional_embedding_size, context_size, worker_values_list,
                                  load_from_context_mode, cfg)

    # Create some more variables for saving purposes
    ini_token_mode_str = ""
    ini_token_from_w = cfg["ini_token_from_w"]
    if ini_token_from_w != 0:
        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"

    correlations_from_weights = cfg["correlations_from_weights"]
    filtering_range = cfg["filtering_range"]

    # Get the requested list of betas
    filtered_beta_list = worker_values_list[min_beta_idx:max_beta_idx]

    show_max_num_patterns = 6  # Just important if we are plotting more than 6 features at the same time

    # If `show_1_feat` is defined it will only plot one feature at a time.
    # The value of the list is the index of the feature to plot.
    show_1_feat = [1, 0, 0]
    # show_1_feat = [None, None, None]
    # Load each stat and plot/save it
    for stat_name in stats_to_save_plot:

        # Create folder if it does not exist and we are saving the image
        if cfg["save_not_plot"] and (not os.path.exists(folder_path + f"/{stat_name}/")):
            os.makedirs(folder_path + f"/{stat_name}/")

        # filter_idx defines what feature we are using for intersecting with 0.
        for filter_idx in range(num_feat_patterns):

            # Title for internal use
            if show_title:
                title = (
                    f"CORRm={correlations_from_weights} CTX={context_size} NUM_PAT={num_feat_patterns} "
                    f"SEED={seed} Filter={filtering_range}")
            else:
                title = None

            # Save path
            filtered_save_path = (folder_path + f"/{stat_name}/seed-" + str(seed) + "-ini_token_idx-" +
                                  str(ini_token_idx) + "-transient_steps-" + str(cfg["num_transient_steps"]) + "-filter_idx-" + str(filter_idx) +
                                  "-filter_rg-" + str(filtering_range) + image_format)

            # Plotting and saving
            print("Creating and saving diagram")
            plot_filtered_bifurcation_diagram_par_imshow(filter_idx, filtered_beta_list, num_feat_patterns,
                                                         filtered_save_path, num_transient_steps_plot_arg,
                                                         stat_name, folder_path, seed, ini_token_idx,
                                                         ini_token_mode_str, filtering_range=filtering_range,
                                                         show_max_num_patterns=show_max_num_patterns,
                                                         save_not_plot=cfg["save_not_plot"], title=title,
                                                         show_1_feat=show_1_feat[filter_idx])

    # For internal use mostly, to decide the final plots. Creates low resolution images of the planes.
    plot_lowres_planes = False
    if plot_lowres_planes:
        for idx in range(len(filtered_beta_list)):

            print(f"Plotting lowres planes for beta {idx + 1}/{len(filtered_beta_list)} ")

            beta_idx = min_beta_idx + idx
            stats_data_path = (folder_path + "/stats" + "/seed-" + str(seed) + "-ini_token_idx-"
                               + str(ini_token_idx) + ini_token_mode_str + "-beta_idx-" + str(beta_idx)
                               + ".npz")

            # Load data
            data = np.load(stats_data_path)
            mo_se_results = data[f"mo_se_results_beta"]
            # 3 feats
            stats_to_plot = [["mo_se"], ["mo_se"]]
            feat_idx = [[0], [1]]

            plot_save_path_plane = (folder_path + f"/indiv_traj_lowres/seed-{str(seed)}/planes"
                                    + f"/plane-beta-{worker_values_list[beta_idx]}-ini_token_idx-" +
                                    str(ini_token_idx) + "-transient_steps-" +
                                    str(cfg["num_transient_steps"]) + image_format)

            if cfg["save_not_plot"]:
                create_dir_from_filepath(plot_save_path_plane)

            stat_results_beta_list_0 = [mo_se_results]
            stat_results_beta_list_1 = [mo_se_results]

            plot_save_plane(stat_results_beta_list_0,
                            stat_results_beta_list_1, cfg["max_sim_steps"] - cfg["num_transient_steps"], feat_idx,
                            tag_names=stats_to_plot, save_path=plot_save_path_plane,
                            save_not_plot=cfg["save_not_plot"], lowres=True)


if __name__ == "__main__":

    # Load cfg
    cfg_path = '../cfgs/bif_diagram_inf_0.yaml'
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    num_bifurcation_values = 4001  # Number of x values to examine in the bifurcation diagram

    worker_values_list = np.linspace(cfg["min_bifurcation_value"], cfg["max_bifurcation_value"],
                                     num_bifurcation_values)  # Betas or Epsilon values

    seed = 1  # List of seeds to review
    num_feat_patterns = 3  # List of number of features for which to initialize the model
    ini_token_idx = 0
    load_from_last_chpt = True # Whether to first simulate the last beta and then simulate the rest from its final context.

    show_title = False  # Whether to plot a title with the characteristics of the experiment. For internal use mostly.

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if cfg["num_transient_steps"] > cfg["max_sim_steps"]:
        raise ("You cannot discard more timesteps than you are simulating.")

    stats_to_save_plot = ["mo_se"]

    start = time.time()

    # Compute the bifurcation diagrams
    if not load_from_last_chpt:
        for worker_id in range(num_bifurcation_values):
            runner(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list,
                   worker_id, cfg, stats_to_save_plot)
    else:
        # First compute the last beta
        runner(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list,
               num_bifurcation_values - 1, cfg, stats_to_save_plot, load_from_context_mode=1)

        # Then compute the rest of the betas, setting the initial context to the last beta one
        for worker_id in range(num_bifurcation_values - 1):
            runner(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list,
                   worker_id, cfg, stats_to_save_plot, load_from_context_mode=2)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    # Once computed, load checkpoints and plot them
    if load_from_last_chpt:
        load_from_context_mode = 1
    else:
        load_from_context_mode = 0
    plotter(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, worker_values_list, cfg,
            stats_to_save_plot, load_from_context_mode=load_from_context_mode, show_title=show_title)
