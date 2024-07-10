#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH="../cfgs/bif_diagram_inf_0.yaml"
NUM_BIFURCATION_VALUES=20

for SEED in "${SEED_LIST[@]}"; do
    for NUM_FEAT_PATTERNS in "${NUM_FEAT_PATTERNS_LIST[@]}"; do
        for POSITIONAL_EMBEDDING_SIZE in "${POSITIONAL_EMBEDDING_SIZE_LIST[@]}"; do
          for INI_TOKEN_IDX in "${INI_TOKEN_IDX_LIST[@]}"; do

                WORKER_ID=$NUM_BIFURCATION_VALUES
                LOAD_FROM_CONTEXT_MODE=1
                echo $SEED $NUM_FEAT_PATTERNS $POSITIONAL_EMBEDDING_SIZE $INI_TOKEN_IDX
                source bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS $POSITIONAL_EMBEDDING_SIZE \
                $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $LOAD_FROM_CONTEXT_MODE $CFG_PATH $WORKER_ID

                LOAD_FROM_CONTEXT_MODE=2
                for WORKER_ID in $(seq 1 $(($NUM_BIFURCATION_VALUES -1))); do
                  echo Num bifurcation values parallel $NUM_BIFURCATION_VALUES $WORKER_ID
                  source bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $LOAD_FROM_CONTEXT_MODE $CFG_PATH  \
                  $WORKER_ID
                done

                LOAD_FROM_CONTEXT_MODE=1
                source bifurcation_diagrams_out_inf_plot.sh $SEED $NUM_FEAT_PATTERNS $POSITIONAL_EMBEDDING_SIZE \
                $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $LOAD_FROM_CONTEXT_MODE $CFG_PATH
          done
        done
    done
done


