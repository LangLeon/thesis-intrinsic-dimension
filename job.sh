#!/bin/bash
#SBATCH -t 00:45:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

python ddim_vs_acc.py --model=lenet --schedule --parameter_correction

