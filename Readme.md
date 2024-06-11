# Dynamical Mean-Field Theory of Self-Attention Neural Networks

This repository contains the code to reproduce the experiments performed in the *Dynamical Mean-Field Theory of Self-Attention Neural Networks*[link to paper] article.

## Abstract

Transformer-based models have demonstrated exceptional performance across diverse domains, becoming the state-of-the-art solution for addressing sequential machine learning problems. Even though we have a general understanding  of the fundamental components in the transformer architecture, little is known about how they operate or what are their expected dynamics. Recently, there has been an increasing interest in exploring the relationship between attention mechanisms and Hopfield networks, promising to shed light on the statistical physics of transformer networks. However, to date, the dynamical regimes of transformer-like models have not been studied in depth. In this paper, we address this gap by using methods for the study of asymmetric Hopfield networks in nonequilibrium regimes --namely path integral methods over generating functionals, yielding dynamics governed by concurrent mean-field variables. Assuming 1-bit tokens and weights, we derive analytical approximations for the behavior of large self-attention neural networks coupled to a softmax output, which become exact in the large limit size. Our findings reveal nontrivial dynamical phenomena, including nonequilibrium phase transitions associated with chaotic bifurcations, even for very simple configurations with a few encoded features and a very short context window. Finally, we discuss the potential of our analytic approach to improve our understanding of the inner workings of transformer models, potentially reducing computational training costs and enhancing model interpretability.

## Installation

In order to install to run the scripts, you must first create and install the requirements in an environment. In order to do so follow the following steps:

1. Create environment if you do not have one: `python -m venv venv`.
2. Load the environment: `source venv/bin/activate`.
3. Install the requirements: `python -m pip install -r requirements.txt`.

## Scripts

### Bifurcation diagrams
To reproduce the bifurcation diagrams in a consumer set-up. Follow the following steps. 

1. First, create or use a config file. You can use templates located in `cfgs` folder. Using `cfgs/bif_diagram_inf_0_zoom-in.yaml` you can replicate the zoomed-in version of the diagram in the paper. Using  `cfgs/bif_diagram_inf_0.yaml` you can replicate the other diagram.  Then, follow one of steps 2 or 3.
2. Configure the bottom part `bifurcation_diagrams.py` and do `python bifurcation_diagrams.py`.
3. Configure `slurm_parallel/launcher_bifurcation_diagrams_inf_ctx_run_localtest.sh` and do `slurm_parallel/source launcher_bifurcation_diagrams_inf_ctx_run_localtest.sh`

If you want to make use of HPC capabilities and compute experiments in parallel using Slurm, first follow step one and then configure and run `launcher_bifurcation_diagrams_inf_ctx_run_zoom-in.sh` or `launcher_bifurcation_diagrams_inf_ctx_run.sh` located in `slurm_parallel`.
Then, to collect the results configure and run `slurm_parallel/launcher_bifurcation_diagrams_inf_ctx_plot.sh`.

### Trajectories.

The calculation of the bifurcation diagrams is very costly computationally. If you want to examine the trajectories presented in the paper, do `python simulate_mf_trajectory_inf_paper_3pats.py`.

