#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH --output ./log_parallel3/exec.%j.out
#SBATCH --error ./log_parallel3/exec.%j.err
# ### SBATCH --output=/dev/null
#SBATCH -N 1 -c 1
#SBATCH -p medium -t 01:30:00
#SBATCH --mem=4G


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
WORKER_ID=$7

if [ -z "$WORKER_ID" ]; # Check if variable is not defined, if not, define it from console args
then
WORKER_ID=$SLURM_ARRAY_TASK_ID
fi

mkdir log_parallel3/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.out

ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH \
--worker_id=$WORKER_ID \
"
echo $ARGS
python bifurcation_diagrams_from_sh_run.py $ARGS > ${log_parallel3}
#python ../bifurcation_diagrams_from_sh_run.py $ARGS