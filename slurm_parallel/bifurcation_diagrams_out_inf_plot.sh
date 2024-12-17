#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH --output ./log_parallel3/exec.%j.out
#SBATCH --error ./log_parallel3/exec.%j.err
#SBATCH -N 1 -c 2
#SBATCH -p short -t 00:30:00
#SBATCH --mem=8G


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


ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--num_bifurcation_values=$NUM_BIFURCATION_VALUES \
--ini_token_idx=$INI_TOKEN_IDX \
--cfg_path=$CFG_PATH \
--exp_dir=$EXP_DIR \
"
echo $ARGS
python bifurcation_diagrams_from_sh_plot.py $ARGS
#python ../bifurcation_diagrams_from_sh_plot.py $ARGS