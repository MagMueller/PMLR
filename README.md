# PMLR
## Setup

```diff 
- The files in file `euler_playground_commands`are a collection of single commands and not runable scripts.
```




### Variables
Add this to your .bashrc in the $HOME dir of the cluster
```bash
vim ~/.bashrc
```
Insert mode with `i` and paste the following:
```bash
module load cuda/12.1.1
module load gcc/8.2.0 python/3.10.4
module proxy
export LD_LIBRARY_PATH=/cluster/apps/nss/gcc-8.2.0/python/3.10.4/x86_64/lib64:$LD_LIBRARY_PATH
```
Close and save with esc and `:wq`.


### Env
Install venv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run gpu interactive session
To run in gpu interactive session:

```bash
srun --gpus=gpu:1 --mem-per-cpu=4G --time=02:00:00 --pty bash -i 
```
You can change to `--gpus=rtx_3090:1` to get more memory and faster gpu.

Now you can access the gpus in the new terminal and activate the venv to run scripts.
```bash
source venv/bin/activate
```

### Activate jupyter notebook execution in the interactive session
Outside of the cluster do: vim ~/.ssh/config and insert:
Change `mmagnus` to your username.

```bash
Host euler.ethz.ch
    HostName euler.ethz.ch
    User mmagnus

Host eu-*
    HostName %h
    ProxyJump euler.ethz.ch
    User mmagnus
``` 


## Troubleshooting
### Internet access in the interactive session
Do in the interactive session:
```bash
module load eth_proxy
```

