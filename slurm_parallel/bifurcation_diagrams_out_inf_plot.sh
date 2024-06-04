#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/TransformerMF
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
INI_TOKEN_IDX=$5
LOAD_FROM_CONTEXT_MODE=$6
CFG_PATH=$7

if [ -z "$WORKER_ID" ]; # Check if variable is not defined
then
WORKER_ID=$SLURM_ARRAY_TASK_ID
fi

ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH \
--load_from_context_mode=$LOAD_FROM_CONTEXT_MODE
"
echo $ARGS
#python bifurcation_diagrams_from_sh_plot.py $ARGS
python ../bifurcation_diagrams_from_sh_plot.py $ARGS