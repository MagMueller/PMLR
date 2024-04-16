#!/bin/bash
#SBATCH --job-name=fourcastnet_gpu_job   # Job name
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --gpus=rtx_3090:1                     # Number of GPUs
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00                  # Time limit hrs:min:sec
#SBATCH --output=fourcastnet_job.log  # Standard output and error log
#SBATCH --error=fourcastnet_job.err  # Standard output and error log

# asdf SBATCH --gres=gpumem:10g
# Memory needed per node
echo "Running as user: $(whoami)"
echo "Primary group: $(id -gn)"
echo "All groups: $(id -Gn)"

module load cuda/12.1.1
cd $SCRATCH/fcn/
singularity exec --nv --bind /cluster:/cluster $SCRATCH/modulus_23.09.sif python simple_inference.py


# singularity exec --nv $HOME/modulus_23.08.sif python simple_inference.py


# srun --pty --gpus=gpu:1  bash
# singularity exec --nv modulus_23.08.sif python simple_inference.py




