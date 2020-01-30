#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

cd $HOME/thesis-intrinsic-dimension

python classify_mnist.py \
	--model=table13slim \
	--N=1 \
	--optimizer=SGD \
	--lr=0.01 \
	--schedule \
	--schedule_gamma=0.3 \
	--schedule_freq=10 \
	--seed=1 \
	--n_epochs=30 \
	--batch_size=64 \
	--d_dim=1000 \
	--print_freq=20 \
	--print_prec=2 \
	#--flips \
        #--subspace-training \
        #--non_wrapped \
        #--chunked \
        #--dense \
        #--parameter_correction
