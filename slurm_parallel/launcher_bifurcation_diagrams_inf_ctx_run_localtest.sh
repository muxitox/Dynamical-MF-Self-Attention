#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH="cfgs/bif_diagram_inf_0.yaml"
NUM_BIFURCATION_VALUES=3


SUFFIX=""
VAR1=$(basename "$PWD")
if [ "$VAR1" = "slurm_parallel" ]; 
then
  SUFFIX="../"
fi

DATE=$(date +%Y%m%d_%H%M%S)/
EXP_DIR_BASE=results_parallel_v3
EXP_DIR=$EXP_DIR_BASE/$DATE/
mkdir -p  ${SUFFIX}${EXP_DIR}/stats/
mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/lyapunov/
mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/planes/



for SEED in "${SEED_LIST[@]}"; do
    for NUM_FEAT_PATTERNS in "${NUM_FEAT_PATTERNS_LIST[@]}"; do
        for POSITIONAL_EMBEDDING_SIZE in "${POSITIONAL_EMBEDDING_SIZE_LIST[@]}"; do
          for INI_TOKEN_IDX in "${INI_TOKEN_IDX_LIST[@]}"; do

                for WORKER_ID in $(seq 1 $(($NUM_BIFURCATION_VALUES))); do
                  echo Num bifurcation values parallel $NUM_BIFURCATION_VALUES $WORKER_ID
                  source slurm_parallel/bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH  \
                  $EXP_DIR_BASE $DATE $WORKER_ID
                done

                source slurm_parallel/bifurcation_diagrams_out_inf_plot.sh $SEED $NUM_FEAT_PATTERNS $POSITIONAL_EMBEDDING_SIZE \
                $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH $EXP_DIR
        done
    done
done


