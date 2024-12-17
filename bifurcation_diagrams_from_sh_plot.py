import numpy as np
import time
from bifurcation_diagrams import plotter
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
parser.add_argument("--ini_token_idx", help="Specify the index of the initial token to choose from.",
                    type=int)
parser.add_argument("--cfg_path", help="Specify path to the YAML config file",
                    type=str)
parser.add_argument("--exp_dir", help="Specify path to where the experiments were saved",
                    type=str)

if __name__ == "__main__":

    args = parser.parse_args()

    cfg_path = args.cfg_path
    # Load cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Load save path from arg vars
    exp_dir = args.exp_dir

    # Instantiate vocabulary vars
    cfg["positional_embedding_size"] = args.positional_embedding_size
    cfg["context_size"] = 2 ** cfg["positional_embedding_size"]

    # Create x values for the bifurcation diagram
    num_bifurcation_values = args.num_bifurcation_values
    worker_values_list = np.linspace(cfg["min_bifurcation_value"], cfg["max_bifurcation_value"],
                                     num_bifurcation_values)  # Betas or Epsilon values

    # Create variables for the Hopfield Transformer (HT)
    cfg["seed"] = args.seed
    cfg["num_feat_patterns"] = args.num_feat_patterns
    cfg["ini_token_idx"] = args.ini_token_idx
    worker_id = args.worker_id - 1  # Subtract 1 for indexing
    show_title = True

    if cfg["context_size"] > 2 ** cfg["positional_embedding_size"]:
        raise ("The positional embedding cannot cover the whole context size.")
    if cfg["num_transient_steps"] > cfg["max_sim_steps"]:
        raise ("You cannot discard more timesteps than you are simulating.")

    stats_to_save_plot = ["mo_se", "att"]

    start = time.time()

    plotter(worker_values_list, cfg, exp_dir, stats_to_save_plot, show_title=show_title)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)
