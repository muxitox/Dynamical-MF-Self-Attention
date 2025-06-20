#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH --output=/dev/null
#SBATCH -N 1 -c 1
#SBATCH -p medium -t 03:00:00
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
EXP_DIR_BASE=$8
DATE=$9
WORKER_ID=${10}

EXP_DIR=$EXP_DIR_BASE/$DATE/


if [ -z "$WORKER_ID" ]; # Check if variable is not defined, if not, define it from console args
then
WORKER_ID=$SLURM_ARRAY_TASK_ID
fi

if [ -z "$SLURM_ARRAY_JOB_ID" ]; # Check if variable is not defined, if not, define it from console args
then
# This is not an array job
LOG_DIR=log/log_parallel_cont/date_${DATE}
LOG_PATH=${LOG_DIR}/${SLURM_JOB_ID}
else
# This is an array job
LOG_DIR=log/log_parallel_cont/date_${DATE}_${SLURM_JOB_ID}_${SLURM_ARRAY_JOB_ID}
LOG_PATH=${LOG_DIR}/${SLURM_ARRAY_TASK_ID}
fi

mkdir -p $LOG_DIR

ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH_PRE \
--exp_dir=$EXP_DIR \
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

echo "It took $((($ENDTIME - $STARTTIME)/60)) minutes to complete this task..." >> ${LOG_PATH}.out

if [[ "$WORKER_ID" -ne 1 ]]; then

   echo "$WORKER_ID" >> ${LOG_PATH}.out
   WORKER_ID=$((WORKER_ID - 1))
   echo "$WORKER_ID" >> ${LOG_PATH}.out

   echo "Queue next experiment with worker ID $WORKER_ID" >> ${LOG_PATH}.out

   sbatch slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE $CFG_PATH_POST  \
                  $EXP_DIR_BASE $DATE $WORKER_ID

else

  echo "We have finished computing the first seed. Now queue the bifurcation diagram in parallel. " >> ${LOG_PATH}.out

  sbatch --array=1-$NUM_BIFURCATION_VALUES slurm_parallel/bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_POST $EXP_DIR_BASE $DATE
fi

