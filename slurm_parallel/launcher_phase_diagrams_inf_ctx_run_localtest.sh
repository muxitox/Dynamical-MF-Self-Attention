#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH="cfgs/phase_diagram_inf_0.yaml"
NUM_VALUES_BETA_ATT=15
NUM_VALUES_BETA_OUT=5
NUM_BIFURCATION_VALUES=$(($NUM_VALUES_BETA_ATT*$NUM_VALUES_BETA_OUT))

for SEED in "${SEED_LIST[@]}"; do
    for NUM_FEAT_PATTERNS in "${NUM_FEAT_PATTERNS_LIST[@]}"; do
        for POSITIONAL_EMBEDDING_SIZE in "${POSITIONAL_EMBEDDING_SIZE_LIST[@]}"; do
          for INI_TOKEN_IDX in "${INI_TOKEN_IDX_LIST[@]}"; do

            for WORKER_ID in $(seq 1 $(($NUM_BIFURCATION_VALUES + 1))); do

                echo Num betas parallel $NUM_BIFURCATION_VALUES
                source slurm_parallel/phase_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $NUM_VALUES_BETA_ATT $NUM_VALUES_BETA_OUT $INI_TOKEN_IDX  \
                $CFG_PATH $WORKER_ID
            done
          done
        done
    done
done


