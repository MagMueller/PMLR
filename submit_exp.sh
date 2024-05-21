# run sbatch run_exp_parallel.sh with model name as argument

models=(deep_coRNN coRNN coRNN2)


wandb_dir="$SCRATCH/wandb"
wandb_cache_dir="$SCRATCH/.cache/wandb"
wandb_config_dir="$SCRATCH/.config/wandb"

# Export the variables
export WANDB_DIR=$wandb_dir
export WANDB_CACHE_DIR=$wandb_cache_dir
export WANDB_CONFIG_DIR=$wandb_config_dir

mkdir -p "$wandb_dir"
mkdir -p "$wandb_cache_dir"
mkdir -p "$wandb_config_dir"


for model in "${models[@]}"
do
    echo "Submitting job for model: $model  ..."
    sbatch --export=ALL run_exp_parallel.sh $model
done
