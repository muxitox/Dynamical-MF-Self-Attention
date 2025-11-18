#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH -N 1 -c 1
#SBATCH -p large -t 20:00:00
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
CFG_PATH=$6
EXP_DIR=$7
WORKER_ID=$8

if [ -z "$WORKER_ID" ]; # Check if variable is not defined, if not, define it from console args
then
WORKER_ID=$SLURM_ARRAY_TASK_ID
fi

# This is not an array job
LOG_DIR=${EXP_DIR}/log/post
LOG_PATH=${LOG_DIR}/${WORKER_ID}

mkdir -p $LOG_DIR

ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH \
--exp_dir=$EXP_DIR \
--worker_id=$WORKER_ID \
"
echo $ARGS >> ${LOG_PATH}.out
python bifurcation_diagrams_from_sh_run.py $ARGS >> ${LOG_PATH}.out 2>> ${LOG_PATH}.err
#python ../bifurcation_diagrams_from_sh_run.py $ARGS

ENDTIME=$(date +%s)
ELAPSEDTIME=$((end - start))
# Convert to minutes (with decimal)
MINUTES=$(echo "scale=2; $ELAPSEDTIME/ 60" | bc)

echo "It took $MINUTES minutes to complete this task..." >> ${LOG_PATH}.out