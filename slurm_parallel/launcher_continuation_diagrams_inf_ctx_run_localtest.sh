#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH_PRE="cfgs/cont_diagram_pre_inf_0_zoom-in.yaml"
CFG_PATH_POST="cfgs/cont_diagram_post_inf_0_zoom-in.yaml"
NUM_BIFURCATION_VALUES=10


SUFFIX=""
VAR1=$(basename "$PWD")
if [ "$VAR1" = "slurm_parallel" ]; 
then
  SUFFIX="../"
fi


EXP_DIR_BASE=results_continuation

for SEED in "${SEED_LIST[@]}"; do
    for NUM_FEAT_PATTERNS in "${NUM_FEAT_PATTERNS_LIST[@]}"; do
        for POSITIONAL_EMBEDDING_SIZE in "${POSITIONAL_EMBEDDING_SIZE_LIST[@]}"; do
          for INI_TOKEN_IDX in "${INI_TOKEN_IDX_LIST[@]}"; do
                DATE=$(date +%Y%m%d_%H%M%S)/
                EXP_DIR=$EXP_DIR_BASE/$DATE/
                mkdir -p  ${SUFFIX}${EXP_DIR}/stats/
                mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/lyapunov/
                mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/planes/
                mkdir -p ${SUFFIX}${EXP_DIR}/lyapunov_traces/

                # Organize it like this so it's similar to the real slurm one
                echo Num bifurcation values parallel $NUM_BIFURCATION_VALUES $NUM_BIFURCATION_VALUES
                source slurm_parallel/bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE  \
                  $EXP_DIR_BASE $DATE $NUM_BIFURCATION_VALUES

                # First pre-compute the initial condition
                for WORKER_ID in $(seq $(($NUM_BIFURCATION_VALUES - 1)) -1 1); do
                  echo Num bifurcation values parallel $NUM_BIFURCATION_VALUES $WORKER_ID
                  source slurm_parallel/bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE  \
                  $EXP_DIR_BASE $DATE $WORKER_ID
                done

                # Now given the initial condition for each beta, compute all values.
                for WORKER_ID in $(seq 1 $(($NUM_BIFURCATION_VALUES))); do
                  echo Num bifurcation values parallel $NUM_BIFURCATION_VALUES $WORKER_ID
                  source slurm_parallel/bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_POST  \
                  $EXP_DIR_BASE $DATE $WORKER_ID
                done

            done
        done
    done
done




# For Hypatia: first see if there's a declared variable with the worker_id. If there is not, assign the last ID,
# execute and at the end sbatch the following one (or end if id is 1)

