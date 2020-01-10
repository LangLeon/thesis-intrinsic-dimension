#!/bin/bash
#SBATCH -t 0:45:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

python ddim_vs_acc.py --model=MLP --schedule

