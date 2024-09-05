import numpy as np
from models.Embedding import Embedding
from models.HopfieldTransformerMFInfNPE import HopfieldTransformerMFInfNPE
from models.HopfieldTransformerMFPE import HopfieldTransformerMFPE
from plotting.plotting import plot_filtered_bifurcation_diagram_par_imshow, get_filtered_values_by_beta_seq
import os
import time
import copy
from utils import create_dir, create_dir_from_filepath
from plotting.plotting import plot_save_plane
import yaml


def create_pathname_inf_betas(num_feat_patterns, positional_embedding_size, context_size, worker_values_list_att,
                              worker_values_list_out, cfg):
    """
    Given the experiment parameters, creates a path to save it.
    The code is a bit intrincate for back-compatibility with older experiments.
    """


    results_folder = f"results_phase/beta_att-beta_o"
    beta_string = (
                "/-min_beta_att-" + str(worker_values_list_att[0]) + "-max_beta_att-" +
                str(worker_values_list_att[-1]) + "-num_betas_att-" + str(len(worker_values_list_att)) +
                "-min_beta_o-" + str(worker_values_list_out[0]) + "-max_beta_o-" +
                str(worker_values_list_out[-1]) + "-num_betas_out-" + str(len(worker_values_list_out)))


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


    # Save/plot results for each ini_token, W config, and num_feat_patterns
    folder_path = (f"{results_folder}/infN-correlations_from_weights-" + str(cfg["correlations_from_weights"])
                   + "-se_size-" + str(cfg["semantic_embedding_size"]) + "-pe_size-"
                   + str(positional_embedding_size) + "-se_per_contribution-" + str(1 - cfg["epsilon_pe"])
                   + "/num_feat_patterns-" + str(num_feat_patterns) + normalize_weights_name_str + scaling_str +
                   compute_inf_normalization_str + "-reorder_weights-" +
                   str(int(cfg["reorder_weights"])) + "-num_segments_corrs-" + str(cfg["num_segments_corrs"])
                   + "-pe_mode-" + str(cfg["pe_mode"]) + gaussian_scale_name_str + "/max_sim_steps-"
                   + str(cfg["max_sim_steps"]) + save_non_transient_str + "-context_size-" + str(context_size)
                   + beta_string)

    return folder_path


def create_pathname(num_feat_patterns, positional_embedding_size, context_size, worker_values_list_att,
                    worker_values_list_out, cfg):

    # Leave this structure to refactor in the future and possibly generalize to other types of phase diagrams
    pathname = create_pathname_inf_betas(num_feat_patterns, positional_embedding_size, context_size,
                                         worker_values_list_att, worker_values_list_out, cfg)

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
    att_window, mv_window, mq_window, mk_window = context_window

    chpt_path = folder_path_chpt + f"/beta_idx-{beta_idx}_window_chpt.npz"

    np.savez_compressed(chpt_path,
                        mv_window=mv_window,
                        mq_window=mq_window,
                        mk_window=mk_window,
                        att_window=att_window)


def load_context(chpt_path):

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


def runner(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, att_values_list,
           out_values_list, worker_id, cfg, stats_to_save_plot):
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
    folder_path = create_pathname(num_feat_patterns, positional_embedding_size, context_size, att_values_list,
                                  out_values_list, cfg)

    chpt_path = cfg["chpt_path"]
    folder_path_stats = folder_path + "/stats"

    # Create folders
    create_dir(folder_path_stats)

    # Define the seed that will create the weights/correlations
    np.random.seed(seed)

    beta_att_idx, beta_out_idx = np.unravel_index(worker_id, (len(att_values_list), len(out_values_list)))

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
    initialize_bifurcation_variable(HT, att_values_list, beta_att_idx, "att")
    initialize_bifurcation_variable(HT, out_values_list, beta_out_idx, "out")


    print(f"Computing seed {seed} beta_att {beta_att_idx} beta_out {beta_out_idx}", flush=True)

    # Reset data structures
    HT.reset_data()

    # Define the initial token. x0 is only used if load_from_context_mode!=2
    x0 = define_ini_token(cfg["ini_token_from_w"], HT, ini_token_idx, ini_tokens_list)
    ini_token_from_w = cfg["ini_token_from_w"]
    if ini_token_from_w != 0:  # Otherwise it's already set
        x0[-positional_embedding_size:] = -1  # Initialize position to -1

    # Load checkpoint from last beta
    mv_window, mq_window, mk_window, att_window = load_context(chpt_path)
    # Set context window to the checkpoint values
    HT.set_context_window(mv_window, mq_window, mk_window, att_window)
    # Simulate from context
    HT.simulate_mf_from_context(max_steps=cfg["max_sim_steps"])

    for stat_name in stats_to_save_plot:
        # Accumulate results in a var of beta_list length
        results_beta[stat_name] = np.copy(HT.mf_statistics[stat_name])

    # Set up some more variables for saving purposes
    ini_token_mode_str = ""
    if ini_token_from_w != 0:
        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"
    stats_data_path = (folder_path_stats + "/seed-" + str(seed) + "-ini_token_idx-" + str(ini_token_idx)
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

    folder_path_unique_points = (folder_path + "/unique_points/seed-" + str(seed)
                                 + "-ini_token_idx-" + str(ini_token_idx) + ini_token_mode_str)


    # Save unique points in mo_se
    unique_data_path = (folder_path_unique_points + "/beta_att-" + str(beta_att_idx) + "-beta_o-" + str(beta_out_idx) + ".npz")
    create_dir(folder_path_unique_points)


    # Set up features for the phase diagram
    feature_to_filter_idx = 0
    filter_by_feature_idx = 1
    min_step_to_look_at = 0
    if cfg["save_non_transient"]:
        min_step_to_look_at = cfg["num_transient_steps"]

    unique_filtered_len, unique_len = get_filtered_values_by_beta_seq(results_beta["mo_se"], feature_to_filter_idx,
                                                                      filter_by_feature_idx, min_step_to_look_at)

    print(unique_filtered_len, unique_len)
    np.savez_compressed(unique_data_path,
                        mo_se_num_filtered_unique=unique_filtered_len,
                        mo_se_num_unique=unique_len,
                        beta_att=att_values_list[beta_att_idx],
                        beta_out=out_values_list[beta_out_idx])

def rejoin_data(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, att_values_list,
                out_values_list, cfg):

    num_betas_att = len(att_values_list)
    num_betas_out = len(out_values_list)

    # Create root folder to later save and aggregate the results
    folder_path = create_pathname(num_feat_patterns, positional_embedding_size, context_size, att_values_list,
                                  out_values_list, cfg)


    ini_token_from_w = cfg["ini_token_from_w"]

    # Set up some more variables for saving purposes
    ini_token_mode_str = ""
    if ini_token_from_w != 0:
        ini_token_mode_str = f"-ini_token_from_w-{ini_token_from_w}"


    unique_points_matrix = np.zeros((num_betas_att, num_betas_out))
    unique_points_filtered_matrix = np.zeros((num_betas_att, num_betas_out))
    for beta_att_idx in range(num_betas_att):
        for beta_out_idx in range(num_betas_out):

            folder_path_unique_points = (folder_path + "/unique_points/seed-" + str(seed)
                                         + "-ini_token_idx-" + str(ini_token_idx) + ini_token_mode_str)
            # Save unique points in mo_se
            unique_data_path = (folder_path_unique_points + "/beta_att-" + str(beta_att_idx) + "-beta_o-" + str(beta_out_idx) + ".npz")

            unique_data = np.load(unique_data_path)

            unique_points_matrix[beta_att_idx][beta_out_idx] = unique_data["mo_se_num_unique"]
            unique_points_filtered_matrix[beta_att_idx][beta_out_idx] = unique_data["mo_se_num_filtered_unique"]



    folder_path_unique_points_agg = (folder_path + "/unique_points_agg/seed-" + str(seed)
                                 + "-ini_token_idx-" + str(ini_token_idx) + ini_token_mode_str)

    create_dir(folder_path_unique_points_agg)
    # Save unique points in mo_se
    unique_data_agg_path = (
                folder_path_unique_points_agg + "/aggretaded_matrix" + ".npz")


    np.savez_compressed(unique_data_agg_path,
                        unique_points_matrix=unique_points_matrix,
                        unique_points_filtered_matrix=unique_points_filtered_matrix)

def plotter(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, num_betas_att,
                num_betas_out, cfg):

    print()

if __name__ == "__main__":

    # Load cfg
    cfg_path = 'cfgs/phase_diagram_inf_0.yaml'
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    num_bifurcation_values_att = 10  # Number of x values to examine in the bifurcation diagram
    num_bifurcation_values_o = 5  # Number of x values to examine in the bifurcation diagram

    att_values_list = np.linspace(cfg["min_bifurcation_value_beta"], cfg["max_bifurcation_value_beta"],
                                     num_bifurcation_values_att)  # Betas or Epsilon values

    out_values_list = np.linspace(cfg["min_bifurcation_value_gamma"], cfg["max_bifurcation_value_gamma"],
                                     num_bifurcation_values_o)  # Betas or Epsilon values

    num_workers = num_bifurcation_values_att * num_bifurcation_values_o

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
    for worker_id in range(num_workers):
        runner(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, att_values_list,
               out_values_list, worker_id, cfg, stats_to_save_plot)


    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)

    # # Once computed, load checkpoints and plot them
    # if load_from_last_chpt:
    #     load_from_context_mode = 1
    # else:
    #     load_from_context_mode = 0
    # plotter(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, num_betas_att, num_betas_out, cfg,
    #         )
