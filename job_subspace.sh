#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --mem=16000M
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

python ddim_vs_acc.py \
	--model=reg_lenet_3 \
	--optimizer=SGD \
	--lr=1 \
	--schedule \
	--schedule_gamma=0.4 \
	--schedule_freq=10 \
	--seed=1 \
	--n_epochs=50 \
	--batch_size=1024 \
	--dense \
#	--non_wrapped \
#	--chunked \
#	--dense \
#	--parameter_correction \
	--print_freq=20 \
	--print_prec=2
