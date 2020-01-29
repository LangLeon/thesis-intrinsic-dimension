#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH -N 1
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --output=slurm_out/test_non_wrapped.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

for N in 2 4 6 8 10 12 14 16
do 
python classify_mnist.py \
	--model=table13slim \
	--N=${N} \
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
	--flips \
        #--subspace-training \
        #--non_wrapped \
        #--chunked \
        #--dense \
        #--parameter_correction
done
