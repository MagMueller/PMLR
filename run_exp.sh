#!/bin/bash
#SBATCH --job-name=corrn_gpu_job   # Job name
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --gpus=gpu:4                     # Number of GPUs
#SBATCH --mem-per-cpu=16G                    # Memory needed per node
#SBATCH --time=10:00:00                  # Time limit hrs:min:sec

#SBATCH --output=log/corrn_job%j.out  # Standard output and error log
#SBATCH --error=log/corrn_job%j.err  # Standard output and error log

module load  cuda/11.8.0 
module load  eth_proxy
module load gcc/9.3.0 python/3.11.2

cd $HOME/PMLR
source pmlr_env/bin/activate

python main.py