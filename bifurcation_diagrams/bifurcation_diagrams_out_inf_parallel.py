import numpy as np
from models.Embedding import Embedding
from models.HopfieldTransformerPEInfN import HopfieldTransformerInfN
from plotting.plotting import plot_filtered_bifurcation_diagram_par_imshow
import os
import time
from utils import create_dir, create_dir_from_filepath
from plotting.plotting import plot_save_plane

def create_pathname_betas(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
                    num_transient_steps, max_sim_steps, context_size, normalize_weights_str_att,
                    normalize_weights_str_o, reorder_weights, se_per_contribution,
                    correlations_from_weights, num_segments_corrs, pe_mode, gaussian_scale_str,
                    save_non_transient, compute_inf_normalization, scaling_o, scaling_att, load_from_context_mode,
                    beta_att=None, beta_o=None, bifurcation_mode="betas"):

    """
    Given the experiment parameters, creates a path to save it
    """

    if bifurcation_mode == "betas":
        results_folder = "results_parallel"
        beta_string = ("/min_beta-" + str(beta_list[0]) + "-max_beta-" + str(beta_list[-1]) +
                       "-num_betas-" + str(len(beta_list)))
    elif bifurcation_mode == "out":
        results_folder = "results_out_parallel"
        beta_string = ("/beta_att-" + str(beta_att) + "-min_beta_o-" + str(beta_list[0]) + "-max_beta_o-" +
                       str(beta_list[-1]) + "-num_betas-" + str(len(beta_list)))
    elif bifurcation_mode == "att":
        results_folder = "results_att_parallel"
        beta_string = ("/beta_o-" + str(beta_o) + "-min_beta_att-" + str(beta_list[0]) + "-max_beta_att-" +
                       str(beta_list[-1]) + "-num_betas-" + str(len(beta_list)))
    else:
        raise Exception("mode not recognized (not one of [\"betas\", \"out\", \"att\", \"pe\"])")

    if correlations_from_weights != 0:
        gaussian_scale_name_str = ""
    else:
        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

    if save_non_transient == True:
        save_non_transient_str = ""
    else:
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"

    if normalize_weights_str_o == normalize_weights_str_att:
        normalize_weights_name_str = "-normalize_weights-" + normalize_weights_str_att
    else:
        normalize_weights_name_str = ("-normalize_weights_att-" + normalize_weights_str_att +
                                      "-normalize_weights_o-" + normalize_weights_str_o)

    scaling_str = ""
    if scaling_o != 1:
        scaling_str += "-scaling_o-" + str(scaling_o)
    if scaling_att != 1:
        scaling_str += "-scaling_att-" + str(scaling_att)

    compute_inf_normalization_str = ""
    if compute_inf_normalization:
        compute_inf_normalization_str = "-inf_norm"

    load_from_context_mode_str = ""
    if load_from_context_mode != 0:
        load_from_context_mode_str = "-load_from_context_mode-1"

    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = (f"{results_folder}/infN-correlations_from_weights-" + str(correlations_from_weights)
                   + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                   + str(positional_embedding_size) + "-se_per_contribution-" + str(se_per_contribution)
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(reorder_weights)) + "-num_segments_corrs-" + str(num_segments_corrs)
                   + "-pe_mode-" + str(pe_mode) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(max_sim_steps) + save_non_transient_str + "-context_size-" + str(context_size)
                   + beta_string + load_from_context_mode_str)

    return folder_path

def create_pathname_pes(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, epsilon_pe_list,
                    num_transient_steps, max_sim_steps, context_size, normalize_weights_str_att,
                    normalize_weights_str_o, reorder_weights, se_per_contribution,
                    correlations_from_weights, num_segments_corrs, pe_mode, gaussian_scale_str,
                    save_non_transient, compute_inf_normalization, scaling_o, scaling_att, load_from_context_mode,
                    beta_att=None, beta_o=None):

    """
    Given the experiment parameters, creates a path to save it
    """

    epsilon_pe_string = ("/min_epsilon_pe-" + str(epsilon_pe_list[0]) + "-min_epsilon_pe-" +
                     str(epsilon_pe_list[-1]) + "-num_pes-" + str(len(epsilon_pe_list)))

    if correlations_from_weights != 0:
        gaussian_scale_name_str = ""
    else:
        gaussian_scale_name_str = f"-gaussian_scale-{gaussian_scale_str}"

    if save_non_transient == True:
        save_non_transient_str = ""
    else:
        save_non_transient_str = f"-num_transient_steps-{num_transient_steps}"

    if normalize_weights_str_o == normalize_weights_str_att:
        normalize_weights_name_str = "-normalize_weights-" + normalize_weights_str_att
    else:
        normalize_weights_name_str = ("-normalize_weights_att-" + normalize_weights_str_att +
                                      "-normalize_weights_o-" + normalize_weights_str_o)

    scaling_str = ""
    if scaling_o != 1:
        scaling_str += "-scaling_o-" + str(scaling_o)
    if scaling_att != 1:
        scaling_str += "-scaling_att-" + str(scaling_att)

    compute_inf_normalization_str = ""
    if compute_inf_normalization:
        compute_inf_normalization_str = "-inf_norm"

    load_from_context_mode_str = ""
    if load_from_context_mode != 0:
        load_from_context_mode_str = "-load_from_context_mode-1"

    if beta_att == beta_o:
        beta_string = "-beta-" + str(beta_att)
    else:
        beta_string = "-beta_o-" + str(beta_o) + "-beta_att-" + str(beta_att)

    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = ("results_out_parallel/infN-correlations_from_weights-" + str(correlations_from_weights)
                   + "-se_size-" + str(tentative_semantic_embedding_size) + "-pe_size-"
                   + str(positional_embedding_size) + beta_string
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(reorder_weights)) + "-num_segments_corrs-" + str(num_segments_corrs)
                   + "-pe_mode-" + str(pe_mode) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(max_sim_steps) + save_non_transient_str + "-context_size-" + str(context_size)
                   + epsilon_pe_string + load_from_context_mode_str)

    return folder_path

def create_pathname(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, worker_values_list,
                    num_transient_steps, max_sim_steps, context_size, normalize_weights_str_att,
                    normalize_weights_str_o, reorder_weights, se_per_contribution,
                    correlations_from_weights, num_segments_corrs, pe_mode, gaussian_scale_str,
                    save_non_transient, compute_inf_normalization, scaling_o, scaling_att, load_from_context_mode,
                    beta_att=None, beta_o=None, bifurcation_mode="betas"):

    if bifurcation_mode == "pe":
        pathname = create_pathname_pes(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size,
                                       worker_values_list, num_transient_steps, max_sim_steps, context_size,
                                       normalize_weights_str_att, normalize_weights_str_o, reorder_weights,
                                       se_per_contribution, correlations_from_weights, num_segments_corrs, pe_mode,
                                       gaussian_scale_str, save_non_transient, compute_inf_normalization,
                                       scaling_o, scaling_att, load_from_context_mode,
                                       beta_att=beta_att, beta_o=beta_o)
    else:
        pathname = create_pathname_betas(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size,
                                         worker_values_list, num_transient_steps, max_sim_steps, context_size,
                                         normalize_weights_str_att, normalize_weights_str_o, reorder_weights,
                                         se_per_contribution, correlations_from_weights, num_segments_corrs, pe_mode,
                                         gaussian_scale_str, save_non_transient, compute_inf_normalization,
                                         scaling_o, scaling_att, load_from_context_mode,
                                         beta_att=None, beta_o=None, bifurcation_mode="betas")

    return pathname


def define_ini_token(ini_token_from_w, HT, ini_token_idx, ini_tokens_list):
    """
    Defines how to set the initial token
    """
    if ini_token_from_w == 0:
        # Encode initial token with position 0
        x0 = ini_tokens_list[ini_token_idx]
    elif ini_token_from_w == 1:
        x0 = HT.Wo[ini_token_idx]
    elif ini_token_from_w == 2:
        x0 = HT.Wv[ini_token_idx]
    elif ini_token_from_w == 3:
        x0 = HT.Wq[ini_token_idx]
    elif ini_token_from_w == 4:
        x0 = HT.Wk[ini_token_idx]
    else:
        raise Exception("ini_token_idx is not in the range [0,4]")

    return x0

def save_context(context_window, folder_path_chpt, beta_idx):
    """
    Saves the mean-field values associated to the context window
    """
    att_window, mv_window, mq_window, mk_window = context_window

    chpt_path = folder_path_chpt + f"/beta_idx-{beta_idx}_window_chpt.npz"

    np.savez_compressed(chpt_path,
                        mv_window=mv_window,
                        mq_window=mq_window,
                        mk_window=mk_window,
                        att_window=att_window)

def load_context(folder_path_chpt, beta_idx):
    """
    Load the mean-field values associated to the context window of a previous experiment.
    :param beta_idx index of the beta from which to load the context window
    """
    chpt_path = folder_path_chpt + f"/beta_idx-{beta_idx}_window_chpt.npz"

    cw = np.load(chpt_path)

    return cw['mv_window'], cw['mq_window'], cw['mk_window'], cw['att_window']

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

def runner(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_o, beta_att,
           num_transient_steps, max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str_att,
           normalize_weights_str_o, reorder_weights, stats_to_save_plot, epsilon_pe,
           correlations_from_weights, num_segments_corrs, pe_mode, gaussian_scale_str,
           save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, worker_values_list,
           worker_id, bifurcation_mode="betas", load_from_context_mode=0):

    """

    :param load_from_context_mode: 0 -> don't load from context, 1 -> don't load from context but save your final context
                                   2-> load context from other experiment
    :return:
    """

    vocab = Embedding(tentative_semantic_embedding_size, positional_embedding_size)

    # Seed equal to 0 for initial token set up
    np.random.seed(0)
    num_ini_tokens = 10  # Number of candidate initial tokens

    ini_tokens_list = np.random.randint(2, size=(
        num_ini_tokens, tentative_semantic_embedding_size + positional_embedding_size)) * 2 - 1
    # Initialize positional embedding
    ini_tokens_list[:, -positional_embedding_size:] = -1


    min_saved_step = 0
    if not save_non_transient:
        min_saved_step = num_transient_steps

    # Create root folder to later save and aggregate the results
    folder_path = create_pathname(num_feat_patterns, tentative_semantic_embedding_size,
                                  positional_embedding_size, worker_values_list, num_transient_steps,
                                  max_sim_steps, context_size, normalize_weights_str_att,
                                  normalize_weights_str_o, reorder_weights, epsilon_pe,
                                  correlations_from_weights, num_segments_corrs, pe_mode,
                                  gaussian_scale_str, save_non_transient,
                                  compute_inf_normalization, scaling_o, scaling_att,
                                  load_from_context_mode, beta_o, beta_att, bifurcation_mode)

    folder_path_chpt = folder_path + "/chpt"
    folder_path = folder_path + "/stats"

    create_dir(folder_path)
    if load_from_context_mode == 1:
        create_dir(folder_path_chpt)

    # Define the seed that will create the weights/correlations
    np.random.seed(seed)

    # Initialize the Hopfield Transformer class. \beta will be set afterwards
    HT = HopfieldTransformerInfN(beta_o, beta_att, num_feat_patterns=num_feat_patterns,
                                 positional_embedding_bitsize=positional_embedding_size, vocab=vocab,
                                 context_size=context_size, max_sim_steps=max_sim_steps,
                                 min_saved_step=min_saved_step,
                                 normalize_weights_str_att=normalize_weights_str_att,
                                 normalize_weights_str_o=normalize_weights_str_o,
                                 reorder_weights=reorder_weights,
                                 correlations_from_weights=correlations_from_weights,
                                 num_segments_corrs=num_segments_corrs, pe_mode=pe_mode,
                                 semantic_embedding_bitsize=tentative_semantic_embedding_size,
                                 se_per_contribution=epsilon_pe,
                                 compute_inf_normalization=compute_inf_normalization,
                                 N_normalization=9999,
                                 scaling_o=scaling_o,
                                 scaling_att=scaling_att)

    # Initialize structure for saving the results for each beta
    results_beta = {}
    for stat_name in HT.statistics_names:
        results_beta[stat_name] = []

    # Set either both betas, one of them or epsilon from the positional encoding
    initialize_bifurcation_variable(HT, worker_values_list, worker_id, bifurcation_mode)

    print(f"Computing seed {seed} beta {worker_id}/{len(beta_list)}", flush=True)

    # Reset data structures
    HT.reset_data()

    # Define the initial token. x0 is only used if load_from_context_mode!=2
    x0 = define_ini_token(ini_token_from_w, HT, ini_token_idx, ini_tokens_list)
    if ini_token_from_w != 0:  # Otherwise it's already set
        x0[-positional_embedding_size:] = -1  # Initialize position to -1

    if load_from_context_mode == 0 or load_from_context_mode == 1:
        # Simulate for max_sim_steps steps from x0
        HT.simulate_mf(x0, max_steps=max_sim_steps)
        if load_from_context_mode == 1:
            # Save context reordered for a fresh start
            HT.reorder_context_window()
            cw = HT.return_context_window()
            save_context(cw, folder_path_chpt, worker_id)
    elif load_from_context_mode == 2:
        # Load checkpoint from last beta
        mv_window, mq_window, mk_window, att_window = load_context(folder_path_chpt, len(beta_list)-1)
        # Set context window to the checkpoint values
        HT.set_context_window(mv_window, mq_window, mk_window, att_window)
        # Simulate from context
        HT.simulate_mf_from_context(max_steps=max_sim_steps)

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


def plotter(num_feat_patterns, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_token_idx, seed, normalize_weights_str_att,
            normalize_weights_str_o, reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights,
            num_segments_corrs, pe_mode, epsilon_pe, gaussian_scale_str,
            save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_o,
            filtering_range, load_from_context_mode=0, min_max_beta_to_show=None, show_title=False):


    # Set up some parameters for loading the experiments statistics
    if min_max_beta_to_show is None:
        min_beta_idx = 0
        max_beta_idx = None
    else:  # In this else, if set, we can zoom_in the bif. diagram but without much resolution
        min_beta_idx = np.searchsorted(beta_list, min_max_beta_to_show[0])
        max_beta_idx = np.searchsorted(beta_list, min_max_beta_to_show[1]) + 1

    if save_non_transient == True:
        num_transient_steps_plot_arg = num_transient_steps
    else:
        num_transient_steps_plot_arg = 0

    # image_format = ".jpeg"
    image_format = ".pdf"


    # Create pathname
    folder_path = create_pathname(num_feat_patterns, tentative_semantic_embedding_size,
                                  positional_embedding_size, beta_list, num_transient_steps,
                                  max_sim_steps, context_size, normalize_weights_str_att,
                                  normalize_weights_str_o, reorder_weights, epsilon_pe,
                                  correlations_from_weights, num_segments_corrs, pe_mode,
                                  gaussian_scale_str, save_non_transient,
                                  compute_inf_normalization, scaling_o, scaling_att, beta_o,
                                  load_from_context_mode)

    # Create some more variables for saving purposes
    ini_token_mode_str = ""
    if ini_token_from_w != 0:
        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"

    # Get the requested list of betas
    filtered_beta_list = beta_list[min_beta_idx:max_beta_idx]

    show_max_num_patterns = 6  # Just important if we are plotting more than 6 features at the same time

    # If `show_1_feat` is defined it will only plot one feature at a time.
    # The value of the list is the index of the feature to plot.
    show_1_feat = [1, 0, 0]
    # show_1_feat = [None, None, None]
    # Load each stat and plot/save it
    for stat_name in stats_to_save_plot:

        # Create folder if it does not exist and we are saving the image
        if save_not_plot and (not os.path.exists(folder_path + f"/{stat_name}/")):
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
                                  str(ini_token_idx) + "-transient_steps-" + str(
                        num_transient_steps) + "-filter_idx-" + str(filter_idx) +
                                  "-filter_rg-" + str(filtering_range) + image_format)

            # Plotting and saving
            print("Creating and saving diagram")
            plot_filtered_bifurcation_diagram_par_imshow(filter_idx, filtered_beta_list, num_feat_patterns,
                                                  filtered_save_path, num_transient_steps_plot_arg,
                                                  stat_name, folder_path, seed, ini_token_idx,
                                                  ini_token_mode_str, filtering_range=filtering_range,
                                                  show_max_num_patterns=show_max_num_patterns,
                                                  save_not_plot=save_not_plot, title=title,
                                                  show_1_feat=show_1_feat[filter_idx])

    # For internal use mostly, to decide the final plots. Creates low resolution images of the planes.
    plot_lowres_planes = False
    if plot_lowres_planes:
        for idx in range(len(filtered_beta_list)):

            print(f"Plotting lowres planes for beta {idx+1}/{len(filtered_beta_list)} ")

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
                                    + f"/plane-beta-{beta_list[beta_idx]}-ini_token_idx-" +
                                    str(ini_token_idx) + "-transient_steps-" +
                                    str(num_transient_steps) + image_format)

            if save_not_plot:
                create_dir_from_filepath(plot_save_path_plane)

            stat_results_beta_list_0 = [mo_se_results]
            stat_results_beta_list_1 = [mo_se_results]

            plot_save_plane(stat_results_beta_list_0,
                            stat_results_beta_list_1, max_sim_steps - num_transient_steps, feat_idx,
                            tag_names=stats_to_plot, save_path=plot_save_path_plane,
                            save_not_plot=save_not_plot, lowres=True)


if __name__ == "__main__":
    # Instantiate vocabulary
    tentative_semantic_embedding_size = 99
    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    # New
    # 1 pat: seed 2 0.25, 0.27
    # 2 pat: seed 18 0.8, 1.3
    # 2 pat: seed 10. pe 2: beta 2.2 - 2.8, pe:4 1.5 - 2.2
    # 3 pat: seed 1 (0.35,0.3) - 0.8, 0.37 - 0.45

    # Create variables for the Hopfield Transformer (HT)
    beta_att = 2.2  # Write it as float
    num_betas = 4001  # Number of betas to examine

    zoom_in = True  # Whether to do the bifurcation diagram of the zoomed in part or not
    if zoom_in:
        beta_list = np.linspace(1.24, 1.28, num_betas)
    else:
        beta_list = np.linspace(0, 3, num_betas)
    epsilon_pe = 0.02

    seed = 1  # List of seeds to review
    num_feat_patterns_list = [3]  # List of number of features for which to initialize the model
    num_transient_steps = 100000
    max_sim_steps = num_transient_steps + 20000

    num_ini_tokens = 1  # Number of initiaal tokens to try for
    ini_token_from_w = 1  # Set the mode of how to set the initial token. 0 random. 1, 2, 3, 4: One of the features of W^{o,v,q,k}
    reorder_weights = False  # Feature not used in the experiments
    normalize_weights_str_o = "N"  # Define the output normalization
    normalize_weights_str_att = "N**2*np.sqrt(M)"  # Defines attentin normalization
    scaling_o = 1
    scaling_att = 100  # beta_att * scaling_att equals gamma in the paper
    filtering_range = 0.001  # Error range for the 0 intersection plane
    compute_inf_normalization = True  # Compute normalization constraints in infinity
    correlations_from_weights = 3  # 0 Use gaussian corrs; 1 Create from weight matrices; 2 Uniform means; 3 Random segments
    gaussian_scale = "0.5"  # Only applicable if correlations_from_weights=0
    pe_mode = 0             # Choose how to initialize the PE. 0 -> set it randomly.
    num_segments_corrs = 3  # Only applicable if correlations_from_weights=3
    save_non_transient = False  # If true, save points from the transient. Not recommended for long trajectories.
    save_not_plot = True    # True save. False plot.
    show_title = False      # Whether to plot a title with the characteristics of the experiment. For internal use mostly.
    load_from_last_chpt = True  # Whether to first simulate the last beta and then simulate the rest from its final context.

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if num_transient_steps > max_sim_steps:
        raise ("You cannot discard more timesteps than you are simulating.")


    bifurcation_var = "betas" # One of ["betas", "out", "att", "pe"]

    stats_to_save_plot = ["mo_se"]

    start = time.time()

    # Compute the bifurcation diagrams
    if not load_from_last_chpt:
        for worker_id in range(num_betas):
            runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
                   num_transient_steps, max_sim_steps, context_size, num_ini_tokens, seed, normalize_weights_str_att,
                   normalize_weights_str_o, reorder_weights, stats_to_save_plot, epsilon_pe,
                   correlations_from_weights, num_segments_corrs, pe_mode, gaussian_scale,
                   save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_att,
                   worker_id)
    else:
        # First compute the last beta
        runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
               num_transient_steps, max_sim_steps, context_size, num_ini_tokens, seed, normalize_weights_str_att,
               normalize_weights_str_o, reorder_weights, stats_to_save_plot, epsilon_pe,
               correlations_from_weights, num_segments_corrs, pe_mode, gaussian_scale,
               save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_att,
               num_betas - 1, load_from_context_mode=1)

        # Then compute the rest of the betas, setting the initial context to the last beta one
        for worker_id in range(num_betas - 1):
            runner(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
                   num_transient_steps, max_sim_steps, context_size, num_ini_tokens, seed, normalize_weights_str_att,
                   normalize_weights_str_o, reorder_weights, stats_to_save_plot, epsilon_pe,
                   correlations_from_weights, num_segments_corrs, pe_mode, gaussian_scale,
                   save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_att,
                   worker_id, load_from_context_mode=2)


    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    # Once computed, load checkpoints and plot them
    ini_tokens_list = range(0, num_ini_tokens)
    if load_from_last_chpt:
        load_from_context_mode = 1
    else:
        load_from_context_mode = 0
    plotter(num_feat_patterns_list, tentative_semantic_embedding_size, positional_embedding_size, beta_list,
            num_transient_steps, max_sim_steps, context_size, ini_tokens_list, seed, normalize_weights_str_att,
            normalize_weights_str_o, reorder_weights, save_not_plot, stats_to_save_plot, correlations_from_weights,
            num_segments_corrs, pe_mode, epsilon_pe, gaussian_scale,
            save_non_transient, compute_inf_normalization, scaling_o, scaling_att, ini_token_from_w, beta_att,
            filtering_range, load_from_context_mode, show_title=show_title)
