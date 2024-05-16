#!/bin/bash
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --time=0-23:59:00
#SBATCH --job-name=corrn_job_long   # Job name
#SBATCH --output=log/corrn_job_long_%j.out  # Standard output and error log
#SBATCH --error=log/corrn_job_long_%j.err  # Standard output and error log
#SBATCH --mem-per-cpu=8G 
module load  cuda/11.8.0 
module load  eth_proxy
module load gcc/9.3.0 python/3.11.2

wandb_dir = $SCRATCH + "/wandb"
wandb_cache_dir = $SCRATCH + "/.cache/wandb"
wandb_config_dir = $SCRATCH + "/.config/wandb"
# export
export WANDB_DIR=$wandb_dir
export WANDB_CACHE_DIR=$wandb_cache_dir
export WANDB_CONFIG_DIR=$wandb_config_dir


cd $HOME/PMLR
source pmlr_env/bin/activate



python main.py --nodes 1 --devices 2
