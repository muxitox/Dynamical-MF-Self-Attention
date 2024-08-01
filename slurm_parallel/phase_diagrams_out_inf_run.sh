#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH --output ./log_parallel/exec.%j.out
#SBATCH --error ./log_parallel/exec.%j.err
#SBATCH -N 1 -c 2
#SBATCH -p short -t 00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=apoc@bcamath.org

#module load Python/3.9.5-GCCcore-10.3.0
#source venv/bin/activate

SEED=$1
NUM_FEAT_PATTERNS=$2
POSITIONAL_EMBEDDING_SIZE=$3
NUM_BIFURCATION_VALUES=$4
NUM_VALUES_BETA_ATT=$5
NUM_VALUES_BETA_OUT=$6
INI_TOKEN_IDX=$7
CFG_PATH=$8
WORKER_ID=$9

if [ -z "$WORKER_ID" ]; # Check if variable is not defined
then
WORKER_ID=$SLURM_ARRAY_TASK_ID
fi

ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--num_values_beta_att=$NUM_VALUES_BETA_ATT \
--num_values_beta_out=$NUM_VALUES_BETA_OUT \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH \
--worker_id=$WORKER_ID \
"
echo $ARGS
python phase_diagrams_from_sh_run.py $ARGS
#python ../bifurcation_diagrams_from_sh_run.py $ARGS