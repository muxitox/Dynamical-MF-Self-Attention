#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH="cfgs/bif_diagram_inf_0_zoom-in.yaml"
NUM_BIFURCATION_VALUES=4001

SUFFIX=""
VAR1=$(basename "$PWD")
if [ "$VAR1" = "slurm_parallel" ]; 
then
  SUFFIX="../"
fi

EXP_DIR_BASE=results_parallel_v3

for SEED in "${SEED_LIST[@]}"; do
    for NUM_FEAT_PATTERNS in "${NUM_FEAT_PATTERNS_LIST[@]}"; do
        for POSITIONAL_EMBEDDING_SIZE in "${POSITIONAL_EMBEDDING_SIZE_LIST[@]}"; do
          for INI_TOKEN_IDX in "${INI_TOKEN_IDX_LIST[@]}"; do

                # Each experiment will be saved in a different folder
                DATE=$(date +%Y%m%d_%H%M%S)/
                EXP_DIR=$EXP_DIR_BASE/$DATE/
                # Create folders here to avoid errors creating them in parallel
                mkdir -p  ${SUFFIX}${EXP_DIR}/stats/
                mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/lyapunov/
                mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/planes/
                mkdir -p ${SUFFIX}${EXP_DIR}/lyapunov_traces/

                echo Num betas parallel $NUM_BIFURCATION_VALUES
                sbatch --array=1-$NUM_BIFURCATION_VALUES bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH $EXP_DIR

                # Sleep to guarantee different folder names
                sleep 5
          done
        done
    done
done


