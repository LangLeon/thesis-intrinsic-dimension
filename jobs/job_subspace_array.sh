#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --mem=16000M
#SBATCH --array=1-32
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl
source activate thesis

HPARAMS_FILE=$HOME/thesis-intrinsic-dimension/jobs/array_job_hyperparameters.txt

cd $HOME/thesis-intrinsic-dimension

python ddim_vs_acc.py \
	--model=table13slim \
	--optimizer=SGD \
	--lr=1 \
	--schedule \
	--schedule_gamma=0.3 \
	--schedule_freq=10 \
	--seed=1 \
	--n_epochs=30 \
	--batch_size=64 \
	--print_freq=20 \
	--print_prec=2 \
	--dense \
	--parameter_correction \
 	$(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
#	--non_wrapped \
#	--chunked \
