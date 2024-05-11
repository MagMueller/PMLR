#!/bin/bash
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --time=0-11:59:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=corrn_job_long   # Job name
#SBATCH --output=log/corrn_job_long%j.out  # Standard output and error log
#SBATCH --error=log/corrn_job_long%j.err  # Standard output and error log

module load  cuda/11.8.0 
module load  eth_proxy
module load gcc/9.3.0 python/3.11.2

cd $HOME/PMLR
source pmlr_env/bin/activate

python main.py --nodes 1 --devices 8