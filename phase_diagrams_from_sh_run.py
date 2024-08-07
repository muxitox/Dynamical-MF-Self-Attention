import numpy as np
import time
from phase_diagrams import runner
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Specify the seed for the RNG", type=int)
parser.add_argument("--num_feat_patterns", help="Specify the number of features/patterns", type=int)
parser.add_argument("--positional_embedding_size", help="Specify the PE size", type=int)
parser.add_argument("--worker_id", help="Specify what value of beta it's going to compute.",
                    type=int)
parser.add_argument("--num_bifurcation_values", help="Specify number x of values in the bifurcation diagram.",
                    type=int)
parser.add_argument("--num_values_beta_att", help="Specify number of beta att  values in the bifurcation diagram.",
                    type=int)
parser.add_argument("--num_values_beta_out", help="Specify number of beta out values in the bifurcation diagram.",
                    type=int)
parser.add_argument("--ini_token_idx", help="Specify the index of the initial token to choose from.",
                    type=int)
parser.add_argument("--cfg_path", help="Specify path to the YAML config file",
                    type=str)


if __name__ == "__main__":

    args = parser.parse_args()

    cfg_path = args.cfg_path
    # Load cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Instantiate vocabulary vars
    positional_embedding_size = args.positional_embedding_size
    context_size = 2 ** positional_embedding_size

    num_bifurcation_values = args.num_bifurcation_values
    num_bifurcation_values_att = args.num_values_beta_att
    num_bifurcation_values_o = args.num_values_beta_out

    att_values_list = np.linspace(cfg["min_bifurcation_value_beta"], cfg["max_bifurcation_value_beta"],
                                  num_bifurcation_values_att)  # Betas or Epsilon values

    out_values_list = np.linspace(cfg["min_bifurcation_value_gamma"], cfg["max_bifurcation_value_gamma"],
                                  num_bifurcation_values_o)  # Betas or Epsilon values

    # Create variables for the Hopfield Transformer (HT)
    seed = args.seed
    num_feat_patterns = args.num_feat_patterns
    ini_token_idx = args.ini_token_idx
    worker_id = args.worker_id - 1  # Subtract for indexing
    show_title = True

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if cfg["num_transient_steps"] > cfg["max_sim_steps"]:
        raise ("You cannot discard more timesteps than you are simulating.")

    stats_to_save_plot = ["mo", "mo_se", "att"]

    start = time.time()

    runner(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, att_values_list,
           out_values_list, worker_id, cfg, stats_to_save_plot)


    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)
