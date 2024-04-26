#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/TransformerMF
#SBATCH --output ./log/exec.%j.out
#SBATCH --error ./log/exec.%j.err
#SBATCH -N 1 -c 2
#SBATCH -p large -t 48:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=apoc@bcamath.org

#module load Python/3.9.5-GCCcore-10.3.0
#source venv/bin/activate

SEED=$1
NUM_FEAT_PATTERNS=$2
NUM_TRANSIENT_STEPS=$3
MAX_SIM_STEPS=$4
COMPUTE_INF_NORMALIZATION=$5
NORMALIZE_WEIGHTS_STR_ATT=$6
NORMALIZE_WEIGHTS_STR_O=${19}
CORRELATIONS_FROM_WEIGHTS=$7
PE_MODE=$8
NUM_SEGMENTS_CORRS=$9
MIN_BETA=${10}
MAX_BETA=${11}
NUM_BETAS=${12}
MIN_PE=${13}
MAX_PE=${14}
NUM_PES=${15}
TENTATIVE_SEMANTIC_EMBEDDING_SIZE=${16}
POSITIONAL_EMBEDDING_SIZE=${17}
PE_FROM_SIZE=${18}


ARGS=" \
--seed=$SEED \
--num_feat_patterns=$NUM_FEAT_PATTERNS \
--num_transient_steps=$NUM_TRANSIENT_STEPS \
--max_sim_steps=$MAX_SIM_STEPS \
--keep_context=True \
--reverse_betas=False \
--reorder_weights=False \
--num_ini_tokens=1 \
--compute_inf_normalization=$COMPUTE_INF_NORMALIZATION \
--normalize_weights_str_att=$NORMALIZE_WEIGHTS_STR_ATT \
--normalize_weights_str_o=$NORMALIZE_WEIGHTS_STR_O \
--correlations_from_weights=$CORRELATIONS_FROM_WEIGHTS \
--pe_mode=$PE_MODE \
--num_segments_corrs=$NUM_SEGMENTS_CORRS \
--save_non_transient=False \
--min_beta=$MIN_BETA \
--max_beta=$MAX_BETA \
--num_betas=$NUM_BETAS \
--min_pe=$MIN_PE \
--max_pe=$MAX_PE \
--num_pes=$NUM_PES \
--tentative_semantic_embedding_size=$TENTATIVE_SEMANTIC_EMBEDDING_SIZE \
--positional_embedding_size=$POSITIONAL_EMBEDDING_SIZE \
--pe_proportion_from_size=$PE_FROM_SIZE \
--save_not_plot=True
"

echo $ARGS
python bifurcation_diagrams_inf_from_sh.py $ARGS
#python ../bifurcation_diagrams_inf_from_sh.py $ARGS