#!/bin/bash
#SBATCH --job-name=fourcastnet_gpu_job   # Job name
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --gpus=2                         # Number of GPUs
#SBATCH --time=02:00:00                  # Time limit hrs:min:sec

# Memory needed per node
#SBATCH --output=fourcastnet_job_%j.log  # Standard output and error log

# Load modules
                  # Latest CUDA version from your list
                 # Python version that might have GPU support

# Assume Singularity is installed, proceed with setting up the container
singularity pull docker://nvcr.io/nvidia/modulus/modulus:23.08
singularity run --nv modulus_23.08.sif 

# Clone and install earth2mip
git clone https://github.com/NVIDIA/earth2mip.git
cd earth2mip && pip install .

# Download and set up necessary files
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/versions/v0.2/files/fcnv2_sm.zip'
unzip fcnv2_sm.zip


# srun --pty --gpus=gpu:1  bash
# singularity exec --nv modulus_23.08.sif python simple_inference.py




