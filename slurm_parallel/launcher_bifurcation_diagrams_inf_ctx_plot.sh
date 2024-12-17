#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH="cfgs/bif_diagram_inf_0.yaml"
NUM_BIFURCATION_VALUES=4001
EXP_DIR=""

for SEED in "${SEED_LIST[@]}"; do
    for NUM_FEAT_PATTERNS in "${NUM_FEAT_PATTERNS_LIST[@]}"; do
        for POSITIONAL_EMBEDDING_SIZE in "${POSITIONAL_EMBEDDING_SIZE_LIST[@]}"; do
          for INI_TOKEN_IDX in "${INI_TOKEN_IDX_LIST[@]}"; do

                source bifurcation_diagrams_out_inf_plot.sh $SEED $NUM_FEAT_PATTERNS $POSITIONAL_EMBEDDING_SIZE \
                $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH
          done
        done
    done
done


