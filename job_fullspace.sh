#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --output=slurm_out/test_non_wrapped.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

python classify_mnist.py \
	--model=lenet \
	--N=16 \
	--flips \
	--optimizer=SGD \
	--lr=0.1 \
	--schedule \
	--schedule_gamma=0.4 \
	--schedule_freq=10 \
	--seed=1 \
	--n_epochs=1 \
	--batch_size=64 \
	--d_dim=1000 \
	--print_freq=20 \
	--print_prec=2 \
        #--subspace-training \
        #--non_wrapped \
        #--chunked \
        #--dense \
        #--parameter_correction
	

