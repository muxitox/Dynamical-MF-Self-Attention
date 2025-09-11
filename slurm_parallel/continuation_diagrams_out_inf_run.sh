#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH --output=/dev/null
#SBATCH -N 1 -c 1
#SBATCH -p short -t 00:30:00
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
DONE_DIR=$9
CHAIN=${10}
WORKER_ID=${11}

if [ -z "$WORKER_ID" ]; # Check if variable is not defined, if not, define it from console args
then
WORKER_ID=$SLURM_ARRAY_TASK_ID
fi

# This is not an array job
LOG_DIR=${EXP_DIR}/log/pre

ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH_PRE \
--exp_dir=$EXP_DIR \
--chain=$CHAIN \
--worker_id=$WORKER_ID \
"
echo $ARGS
python bifurcation_diagrams_from_sh_run.py $ARGS
EXIT_CODE=$?
#python ../bifurcation_diagrams_from_sh_run.py $ARGS

if [[ "EXIT_CODE" -ne 0 ]]; then
  echo "Error in the execution. Exiting..."
  exit 1
fi

ENDTIME=$(date +%s)
ELAPSEDTIME=$((end - start))
# Convert to minutes (with decimal)
MINUTES=$(echo "scale=2; $ELAPSEDTIME/ 60" | bc)

echo "It took $MINUTES minutes to complete this task..."

if [[ "$CHAIN" == "0" ]]; then
  echo "Central job finished. Slurm will now be able to start computing the left and right chains."
  exit 0
fi

# If chain==0, you don't get to execute the part below

if [[ $WORKER_ID -lt $NUM_BIFURCATION_VALUES && "$CHAIN" == "+1" ]] || [[ $WORKER_ID -gt 1 && "$CHAIN" == "-1" ]]; then

  echo $CHAIN $WORKER_ID
  if [[ $WORKER_ID -lt $NUM_BIFURCATION_VALUES && "$CHAIN" == "+1" ]]; then
      echo "DEBUG: first condition is TRUE"
  fi

  if [[ $WORKER_ID -lt $NUM_BIFURCATION_VALUES ]]; then
      echo "$WORKER_ID -lt $NUM_BIFURCATION_VALUES"
  fi

  if [[ "$CHAIN" == "+1" ]]; then
      echo "$CHAIN" == "+1"
  fi

  if [[ $WORKER_ID -gt 1 && "$CHAIN" == "-1" ]]; then
      echo "DEBUG: second condition is TRUE"
  fi

  if [[ $WORKER_ID -gt 1 ]]; then
      echo "$WORKER_ID -gt 1"
  fi

  if [[ "$CHAIN" == "-1" ]]; then
      echo "$CHAIN" == "-1"
  fi

   # If the worker index is neither at the beginning or the end of the chain, queue a new job
   WORKER_ID=$((WORKER_ID + CHAIN))

   LOG_PATH=/home/apoc/projects/Dynamical-MF-Self-Attention/${LOG_DIR}/${WORKER_ID}

   sbatch --output=$LOG_PATH.out \
          --error=$LOG_PATH.err \
          slurm_parallel/continuation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_PRE $CFG_PATH_POST  \
                  $EXP_DIR $DONE_DIR $CHAIN $WORKER_ID

else

  echo "We have finished computing the $CHAIN chain."

  touch "$DONE_DIR/$CHAIN.done"
  LOCK=$DONE_DIR/collector.lock


  if [[ -f "$DONE_DIR/-1.done" && -f "$DONE_DIR/+1.done" ]]; then
    if (set -o noclobber; echo $$ > "$LOCK") 2>/dev/null; then
        echo "Both chains finished. Collector can be launched"
        sbatch --array=1-$NUM_BIFURCATION_VALUES \
                  slurm_parallel/bifurcation_diagrams_out_inf_run.sh $SEED $NUM_FEAT_PATTERNS \
                  $POSITIONAL_EMBEDDING_SIZE $NUM_BIFURCATION_VALUES $INI_TOKEN_IDX $CFG_PATH_POST $EXP_DIR
    else
        echo "Waiting for the other chain to finish"
    fi
  fi
fi
