
#!/bin/bash
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --time=0-02:00:00
#SBATCH --job-name=corrn_job   # Job name
#SBATCH --output=log/corrn_job%j.out  # Standard output and error log
#SBATCH --error=log/corrn_job%j.err  # Standard output and error log

module load  cuda/11.8.0 
module load  eth_proxy
module load gcc/9.3.0 python/3.11.2

cd $HOME/PMLR
source pmlr_env/bin/activate

python main.py