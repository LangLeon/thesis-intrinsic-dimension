#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --mem=16000M
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

python classify_mnist.py \
	--model=lenet \
	--optimizer=SGD \
#	--subspace-training \
	--lr=0.1 \
	--schedule \
	--schedule_gamma=0.4 \
	--schedule_freq=10 \
	--seed=1 \
	--n_epochs=30 \
	--batch_size=64 \
	--d_dim=1000 \
#	--non_wrapped \
#	--chunked \
#	--dense \
#	--parameter_correction \
	--print_freq=20 \
	--print_prec=2
	

