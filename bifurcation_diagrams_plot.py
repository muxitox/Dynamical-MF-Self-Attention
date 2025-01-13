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

    # Load save path from arg vars
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        exp_dir = "results_parallel_v3/old_20241217_170124"
        # exp_dir = "results_parallel_v3/old_20241217_170128_zoom"


    cfg_path = exp_dir + "/cfg.yaml"
    # Load cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Create x values for the bifurcation diagram. 
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
