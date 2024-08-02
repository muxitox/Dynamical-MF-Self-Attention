import yaml
import numpy as np
from phase_diagrams import rejoin_data

if __name__ == "__main__":

    # Load cfg
    cfg_path = 'cfgs/phase_diagram_inf_0.yaml'
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    positional_embedding_size = 2
    context_size = 2 ** positional_embedding_size

    num_bifurcation_values_att = 15  # Number of x values to examine in the bifurcation diagram
    num_bifurcation_values_o = 5  # Number of x values to examine in the bifurcation diagram

    att_values_list = np.linspace(cfg["min_bifurcation_value_beta"], cfg["max_bifurcation_value_beta"],
                                     num_bifurcation_values_att)  # Betas or Epsilon values

    out_values_list = np.linspace(cfg["min_bifurcation_value_gamma"], cfg["max_bifurcation_value_gamma"],
                                     num_bifurcation_values_o)  # Betas or Epsilon values

    num_workers = num_bifurcation_values_att * num_bifurcation_values_o

    seed = 1  # List of seeds to review
    num_feat_patterns = 3  # List of number of features for which to initialize the model
    ini_token_idx = 0

    show_title = False  # Whether to plot a title with the characteristics of the experiment. For internal use mostly.

    if context_size > 2 ** positional_embedding_size:
        raise ("The positional embedding cannot cover the whole context size.")
    if cfg["num_transient_steps"] > cfg["max_sim_steps"]:
        raise ("You cannot discard more timesteps than you are simulating.")

    rejoin_data(num_feat_patterns, seed, positional_embedding_size, context_size, ini_token_idx, att_values_list,
                out_values_list, cfg)
