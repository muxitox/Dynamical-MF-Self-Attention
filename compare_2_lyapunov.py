import numpy as np
import os
from plotting.plotting import plot_save_statistics, plot_save_fft, plot_save_plane, plot_lyapunov_graphs
import matplotlib.pyplot as plt
from utils import create_dir_from_filepath, load_lyapunov
import yaml

if __name__ == "__main__":
    exp_dir_1 = "results_continuation/20250704_160253_zoom-2_right"
    exp_dir_2 = "results_continuation/20250925_171913_zoom2_middle2"


    # Load 1st exp
    cfg_path_1 = exp_dir_1 + "/cfg.yaml"
    # Load cfg
    with open(cfg_path_1, 'r') as file:
        cfg_1 = yaml.safe_load(file)

    # Create x values for the bifurcation diagram.
    x_list = np.linspace(cfg_1["min_bifurcation_value"], cfg_1["max_bifurcation_value"],
                                     cfg_1["num_bifurcation_values"])  # Betas or Epsilon values

    # Load 2nd exp
    cfg_path_2 = exp_dir_2 + "/cfg.yaml"
    # Load cfg
    with open(cfg_path_2, 'r') as file:
        cfg_2 = yaml.safe_load(file)

    # Create x values for the bifurcation diagram.
    x_list_2 = np.linspace(cfg_2["min_bifurcation_value"], cfg_2["max_bifurcation_value"],
                                     cfg_2["num_bifurcation_values"])  # Betas or Epsilon values

    if not np.allclose(x_list, x_list_2):
        raise Exception("x_list != x_list_2. "
                        "The two experiments do not have the same domain to be compared")

    num_feat_patterns = cfg_1["num_feat_patterns"]
    context_size = cfg_1["context_size"]

    # Load stats from exps
    valid_S_1, num_valid_dims_1 = load_lyapunov(exp_dir_1, num_feat_patterns, context_size, x_list, 0)
    valid_S_2, num_valid_dims_2 = load_lyapunov(exp_dir_2, num_feat_patterns, context_size, x_list, 0)

    # If the leftmost x value is 0, in case of being a bifurcation or continuation diagrams of \beta we don't want
    # to process this value since the lyapunov exponents computation makes no sense (all ms are 0)
    if x_list[0] == 0.0:
        x_list = x_list[1:]
        valid_S_1 = valid_S_1[1:]
        valid_S_2 = valid_S_2[1:]

    plt.plot(x_list, valid_S_1[:,0], "tab:blue", ls="--", alpha=0.3, label="right_left")
    plt.plot(x_list, valid_S_2[:,0], "tab:blue", label="center")

    plt.plot(x_list, valid_S_1[:, 1], "tab:orange", ls="--", alpha=0.3,)
    plt.plot(x_list, valid_S_2[:, 1], "tab:orange")
    plt.legend()

    plt.show()

    # plt.plot(valid_S_1[:,0], x_list, "blue", label="1")

