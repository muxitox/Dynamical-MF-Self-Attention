#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -D /home/apoc/projects/Dynamical-MF-Self-Attention
#SBATCH --output ./log_parallel_phase_2/exec.%j.out
#SBATCH --error ./log_parallel_phase_2/exec.%j.err
#SBATCH -N 1 -c 1
#SBATCH -p short -t 00:10:00
#SBATCH --mem=8G

module load Python/3.9.5-GCCcore-10.3.0
source venv/bin/activate

python phase_diagram_analysis.py $ARGS
#python ../bifurcation_diagrams_from_sh_run.py $ARGS