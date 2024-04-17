#!/bin/bash
#SBATCH --job-name=fourcastnet_gpu_job   # Job name
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --gpus=gpu:1                     # Number of GPUs
#SBATCH --mem-per-cpu=4G                    # Memory needed per node
#SBATCH --time=02:00:00                  # Time limit hrs:min:sec

#SBATCH --output=fourcastnet_job.log  # Standard output and error log
#SBATCH --error=fourcastnet_job.err  # Standard output and error log

cd $HOME/PMLR
source venv/bin/activate

python test_cuda.py