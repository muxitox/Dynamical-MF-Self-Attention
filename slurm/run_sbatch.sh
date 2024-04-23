#!/bin/bash
#SBATCH --job-name="transformer-mf"
#SBATCH -J serialjob
#SBATCH -D /home/apoc/projects/TransformerMF
#SBATCH --output ./log/exec.%j.out
#SBATCH --error ./log/exec.%j.err
#SBATCH -N 1
#SBATCH -p short -t 00:20:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=apoc@bcamath.org

module load Python/3.9.5-GCCcore-10.3.0
source venv/bin/activate

python simulate_mf_trajectory_inf.py