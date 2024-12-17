#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH="cfgs/bif_diagram_inf_0.yaml"
NUM_BIFURCATION_VALUES=10
EXP_DIR=results_parallel_v3/$(date +%Y%m%d_%H%M%S)

mkdir $EXP_DIR
mkdir $EXP_DIR/stats/
mkdir -p $EXP_DIR/indiv_lowres_traj/lyapunov/
mkdir -p $EXP_DIR/indiv_lowres_traj/planes/

for SEED in "${SEED_LIST[@]}"; do
    for NUM_FEAT_PATTERNS in "${NUM_FEAT_PATTERNS_LIST[@]}"; do
        for POSITIONAL_EMBEDDING_SIZE in "${POSITIONAL_EMBEDDING_SIZE_LIST[@]}"; do
          for INI_TOKEN_IDX in "${INI_TOKEN_IDX_LIST[@]}"; do

                echo Num betas parallel $NUM_BIFURCATION_VALUES
                sbatch --array=1-$NUM_BIFURCATION_VALUES bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH
          done
        done
    done
done


