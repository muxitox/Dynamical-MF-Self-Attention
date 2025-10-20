#!/bin/bash

SEED_LIST=(1)
NUM_FEAT_PATTERNS_LIST=(3)
POSITIONAL_EMBEDDING_SIZE_LIST=(2)
INI_TOKEN_IDX_LIST=(0)
CFG_PATH_PRE="cfgs/cont_diagram_pre_inf_0.yaml"
CFG_PATH_POST="cfgs/cont_diagram_post_inf_0.yaml"
NUM_BIFURCATION_VALUES=501
INI_WORKER_ID=501 # Number between 1 and NUM_BIFURCATION_VALUES

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
                DONE_DIR=${SUFFIX}${EXP_DIR}/job_done


                # Create folders here to avoid errors creating them in parallel
                mkdir -p  ${SUFFIX}${EXP_DIR}/stats/
                mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/lyapunov/
                mkdir -p ${SUFFIX}${EXP_DIR}/indiv_lowres_traj/planes/
                mkdir -p ${SUFFIX}${EXP_DIR}/lyapunov_traces/
                mkdir -p $DONE_DIR

                LOG_DIR=${EXP_DIR}/log/pre
                mkdir -p $LOG_DIR

                LOG_PATH=/home/apoc/projects/Dynamical-MF-Self-Attention/${LOG_DIR}/${INI_WORKER_ID}

                # First submit the initial job
                CHAIN="0"
                echo Log path for the first job: $LOG_PATH

                echo Num bifurcation values parallel $NUM_BIFURCATION_VALUES $NUM_BIFURCATION_VALUES
                jobid=$(sbatch --output=$LOG_PATH.out \
                                --error=$LOG_PATH.err \
                                slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                                $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE  \
                                $CFG_PATH_POST $EXP_DIR $DONE_DIR $CHAIN $INI_WORKER_ID | awk '{print $4}')

                echo We will create a dependency on job $jobid to finish

                # Save this into a config file
                printf "INI_WORKER_ID:\t%s\n" "$INI_WORKER_ID" >> "${SUFFIX}${EXP_DIR}/ini_worker_cfg.yaml"

                # The dependency argument does not work well with the script arguments and the #SBATCH directives
                # So we'll include them in the call

                # Then submit the left and right chains. -1 LEFT, +1 RIGHT

                # If initial ID is already 1, then don't submit and create done file to organize the final collection
                CHAIN="-1"
                if [[ "$INI_WORKER_ID" -ne 1 ]]; then

                  WORKER_ID_L=$((INI_WORKER_ID - 1))
                  echo Queuing chain start [$WORKER_ID_L - 1]

                  LOG_PATH=/home/apoc/projects/Dynamical-MF-Self-Attention/${LOG_DIR}/${WORKER_ID_L}

                  sbatch -D /home/apoc/projects/Dynamical-MF-Self-Attention/ \
                        --output=$LOG_PATH.out \
                        --error=$LOG_PATH.err \
                        -N 1 -c 1 \
                        -p short -t 00:30:00 \
                        --mem=4G \
                        --dependency=afterok:$jobid \
                        slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                        $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE  \
                        $CFG_PATH_POST $EXP_DIR $DONE_DIR $CHAIN $WORKER_ID_L
                else
                  echo Creating lock file "$DONE_DIR/$CHAIN.done"
                  touch "$DONE_DIR/$CHAIN.done"

                fi


                # If initial ID is already NUM_BIFURCATION_VALUES, then don't submit and create lock file to organize the final collection
                CHAIN="+1"
                if [[ "$INI_WORKER_ID" -ne "$NUM_BIFURCATION_VALUES" ]]; then

                  WORKER_ID_R=$((INI_WORKER_ID + 1))
                  echo Queuing chain start [$WORKER_ID_R - $NUM_BIFURCATION_VALUES]

                  LOG_PATH=/home/apoc/projects/Dynamical-MF-Self-Attention/${LOG_DIR}/${WORKER_ID_R}

                  sbatch -D /home/apoc/projects/Dynamical-MF-Self-Attention \
                        --output=$LOG_PATH.out \
                        --error=$LOG_PATH.err \
                        -N 1 -c 1 \
                        -p short -t 00:30:00 \
                        --mem=4G \
                        --dependency=afterok:$jobid \
                        slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                        $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE  \
                        $CFG_PATH_POST $EXP_DIR $DONE_DIR $CHAIN $WORKER_ID_R


                  else

                  echo Creating lock file "$DONE_DIR/$CHAIN.done"
                  touch "$DONE_DIR/$CHAIN.done"

                fi
            done
        done
    done
done



