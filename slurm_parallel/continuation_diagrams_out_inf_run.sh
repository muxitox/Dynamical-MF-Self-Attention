#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH --output=/dev/null
#SBATCH -N 1 -c 1
#SBATCH -p medium -t 04:30:00
#SBATCH --mem=4G


STARTTIME=$(date +%s)

# If we are in CentOS (Hypatia HPC) load modules
if lsb_release -ar 2>/dev/null | grep -q CentOS
then
  module load Python/3.9.5-GCCcore-10.3.0
  source venv/bin/activate
fi

SEED=$1
NUM_FEAT_PATTERNS=$2
POSITIONAL_EMBEDDING_SIZE=$3
NUM_BIFURCATION_VALUES=$4
INI_TOKEN_IDX=$5
CFG_PATH_PRE=$6
CFG_PATH_POST=$7
EXP_DIR=$8
CONT_ORDER=$9
WORKER_ID=${10}

if [ -z "$WORKER_ID" ]; # Check if variable is not defined, if not, define it from console args
then
WORKER_ID=$SLURM_ARRAY_TASK_ID
fi

# This is not an array job
LOG_DIR=${EXP_DIR}/log/pre
LOG_PATH=${LOG_DIR}/${WORKER_ID}

mkdir -p $LOG_DIR

ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH_PRE \
--exp_dir=$EXP_DIR \
--cont_order=$CONT_ORDER \
--worker_id=$WORKER_ID \
"
echo $ARGS >> ${LOG_PATH}.out
python bifurcation_diagrams_from_sh_run.py $ARGS >> ${LOG_PATH}.out 2>> ${LOG_PATH}.err
EXIT_CODE=$?
#python ../bifurcation_diagrams_from_sh_run.py $ARGS

if [[ "EXIT_CODE" -ne 0 ]]; then
  echo "Error in the execution. Exiting..." >> ${LOG_PATH}.out
  exit 1
fi

ENDTIME=$(date +%s)
ELAPSEDTIME=$((end - start))
# Convert to minutes (with decimal)
MINUTES=$(echo "scale=2; $ELAPSEDTIME/ 60" | bc)

echo "It took $MINUTES minutes to complete this task..." >> ${LOG_PATH}.out


if  [[ "$CONT_ORDER" == "l" ]] && [[ "$WORKER_ID" -ne 1 ]]; then

   WORKER_ID=$((WORKER_ID - 1))

   echo "Queue next experiment with worker ID $WORKER_ID" >> ${LOG_PATH}.out

   sbatch slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE $CFG_PATH_POST  \
                  $EXP_DIR $CONT_ORDER $WORKER_ID

elif [[ "$CONT_ORDER" == "r" ]] && [[ "$WORKER_ID" -ne $NUM_BIFURCATION_VALUES ]]; then

  WORKER_ID=$((WORKER_ID + 1))

   echo "Queue next experiment with worker ID $WORKER_ID" >> ${LOG_PATH}.out

   sbatch slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE $CFG_PATH_POST  \
                  $EXP_DIR $CONT_ORDER $WORKER_ID

elif [[ "$CONT_ORDER" == "c" ]] && [[ "$WORKER_ID" -ne $NUM_BIFURCATION_VALUES ]]; then

   WORKER_ID_L=$((WORKER_ID - 1))
   WORKER_ID_R=$((WORKER_ID + 1))

   echo "Queue next experiment with worker ID $WORKER_ID" >> ${LOG_PATH}.out

   sbatch slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE $CFG_PATH_POST  \
                  $EXP_DIR "r" $WORKER_ID_R

   sbatch slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE $CFG_PATH_POST  \
                  $EXP_DIR "l" $WORKER_ID_L

else

  echo "We have finished computing the first seed. Now queue the bifurcation diagram in parallel. " >> ${LOG_PATH}.out

  sbatch --array=1-$NUM_BIFURCATION_VALUES slurm_parallel/bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_POST $EXP_DIR
fi

#TODO:  change python code to assimilate the cont_order parameter.
# TODO: check if there's any argument in this slurm version that allows me to wait for all jobs to finish given that they all have different ids