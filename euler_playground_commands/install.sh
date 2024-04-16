#!/bin/bash
#SBATCH --job-name=fourcastnet_gpu_job   # Job name
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --gpus=gpu:1                     # Number of GPUs
#SBATCH --mem-per-cpu=4G                    # Memory needed per node
#SBATCH --time=02:00:00                  # Time limit hrs:min:sec
#SBATCH --output=fourcastnet_job.log  # Standard output and error log
#SBATCH --error=fourcastnet_job.err  # Standard output and error log



# IMPORTANT: After 2 weeks the files in the scratch directory will be deleted, so save your progress to github.
# IMPORTANT: This installation is not needed for the jupyter notebook, but just if you want to install modulus
# IMPORTANT: Copy paste the commands one by one to the terminal this is not yet completley automated, use your brain:)


# go to scratch because more storage
cd $SCRATCH
singularity pull docker://nvcr.io/nvidia/modulus/modulus:23.08

# mkdir  $SCRATCH/models
mkdir  $SCRATCH/models

# Load cuda and python
module load cuda/12.1.1
module load gcc/8.2.0 python/3.10.4

# run interactive session with gpu
srun --pty --gpus=gpu:1 --mem-per-cpu=4G --time=02:00:00 bash -i
# Alternative with 3090
# srun --gpus=rtx_3090:1 --mem-per-cpu=8G --time=02:00:00 --pty bash -i

# this needs gpu hardware with --nv 
singularity run --nv --compat --pwd $SCRATCH --bind /cluster:/cluster $SCRATCH/modulus_23.08.sif 

# Clone and install earth2mip
git clone https://github.com/NVIDIA/earth2mip.git
cd earth2mip && pip install .

# go to models
cd $SCRATCH/models

# Download and set up necessary files
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/versions/v0.2/files/fcnv2_sm.zip'
unzip fcnv2_sm.zip

# make sure to have .cdsapirc in home directory otherwise install install https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key

# run python script
python fcnv2_sm/simple_inference.py



# srun --pty --gpus=gpu:1 --mem-per-cpu=4G --time=02:00:00 bash -i
# singularity exec --nv modulus_23.08.sif python simple_inference.py




