import numpy as np
import time
from bifurcation_diagrams import plotter
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", help="Specify path to where the experiments were saved.",
                    type=str)

if __name__ == "__main__":

    args = parser.parse_args()

    cfg_path = args.exp_dir
    # Load cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Load save path from arg vars
    exp_dir = args.exp_dir

    # Create x values for the bifurcation diagram. 
    cfg["num_bifurcation_values"] = args.num_bifurcation_values # Save in cfg for replication purposes
    worker_values_list = np.linspace(cfg["min_bifurcation_value"], cfg["max_bifurcation_value"],
                                     cfg["num_bifurcation_values"])  # Betas or Epsilon values

    show_title = True

    stats_to_save_plot = ["mo_se", "att"]

    start = time.time()

    plotter(worker_values_list, cfg, exp_dir, stats_to_save_plot, show_title=show_title)

    end = time.time()
    elapsed_time = end - start
    print("elapsed time in minutes", elapsed_time / 60)
    print("elapsed time in hours", elapsed_time / 3600)
