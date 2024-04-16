#singularity run --nv modulus_23.08.sif
#singularity run --nv -B $HOME:/home_in_container -B $SCRATCH:/scratch_in_container modulus_23.08.sif
#singularity exec --nv modulus_23.08.sif python simple_inference.py
#apptainer shell --nv --bind /cluster:/cluster $SCRATCH/modulus_23.08.sif
cd $SCRATCH/earth2mip/fcnv2_sm/
#module spider  python_gpu/3.7.4

module load cuda/12.1.1
module load gcc/8.2.0 python/3.10.4

export LD_LIBRARY_PATH=/cluster/apps/nss/gcc-8.2.0/python/3.10.4/lib:$LD_LIBRARY_PATH
singularity exec --nv --bind /cluster:/cluster $SCRATCH/modulus_23.08.sif python simple_inference.py
singularity run --nv --compat --bind /cluster:/cluster $SCRATCH/modulus_23.08.sif



srun --gpus=rtx_3090:1 --mem-per-cpu=8G --time=02:00:00 --pty bash -i 

singularity run --nv --compat --pwd $SCRATCH/models --bind /cluster:/cluster $SCRATCH/modulus_23.08.sif
python fcnv2_sm/simple_inference.py
nvcc --version
#  srun --gpus=rtx_3090:1 --mem-per-cpu=8G --time=02:00:00 --pty bash -i
# source venv/bin/activate
#python simple_inference.py
# srun --pty --gpus=gpu:1  bash

