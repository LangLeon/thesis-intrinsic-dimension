#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
conda activate thesis

srun python ddim_vs_acc.py --model=MLP
srun python ddim_vs_acc.py --model=lenet

