# PMLR
## Setup

You dont nessesarily need to login to the cluster to develop your models. You can run them locally on your machine. In your normal setup.
Its just useful for training.

### Login to cluster
Be in ETH network or use VPN. Change `mmagnus` to your username. 
```bash
ssh mmagnus@euler.ethz.ch
```
Enter the password.
If you have troubles with that step, check out: https://scicomp.ethz.ch/wiki/Login_Nodes 


### Variables
Add this to your .bashrc in the $HOME dir of the cluster. Do:
```bash
vim ~/.bashrc
```
Insert mode with `i` and paste the following this will make sure that we use the same python version and cuda version.:
```bash
module load cuda/12.1.1
module load gcc/8.2.0 python/3.10.4
module proxy
export LD_LIBRARY_PATH=/cluster/apps/nss/gcc-8.2.0/python/3.10.4/x86_64/lib64:$LD_LIBRARY_PATH
```
Close and save with esc and `:wq`. Restart vs-code and login again.

### Space on cluster
Check the space on the cluster with:
```bash
lquota
```
`$SCRATCH`: Here you have much space, e.g. for our big datasets, but after 2 weeks the files in the $SCRATCH directory will be deleted, so save your code progress to github.

`$HOME`: Here you have less space, but the files will not be deleted.


### Clone the repo
Set git user and email:
```bash
git config --global user.email "example@yourgithubemail.com"
git config --global user.name "yourgithubusername"
```

```bash
cd $HOME
git clone https://github.com/MagMueller/PMLR.git
cd $HOME/PMLR
```

### Env
Install venv:
```bash
cd $HOME/PMLR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
When you need more packages, install them with pip and then do:
```bash
pip freeze > requirements.txt
```
So that we all have the same packages.

## Run scripts
### Run with gpu in interactive session to debug
To run in gpu interactive session:

```bash
srun --gpus=gpu:1 --mem-per-cpu=4G --time=02:00:00 --pty bash -i 
```
You can change to `--gpus=rtx_3090:1` to get more memory and faster gpu.
Wait until resources are allocated and you are in the interactive session.

Now you can access the gpus in the new terminal and activate the venv to run scripts.
```bash
cd $HOME/PMLR
source venv/bin/activate
```

Run test script to check if cuda is available:
```bash
python test_cuda.py
```
It should print that cuda is available.
If you run it on the normal login node (not in the interactive session) it will say `cuda available: False`, because you dont have access to the gpus.

### Submit job to cluster to train
To submit a job to the cluster i provided a sample script:
```bash
sbatch euler_playground_commands/example_script_submit_job.sh
```
Check the status of the job with:
```bash
squeue 
```
And the output of the job with can be found in `.log`and `.err`files.



## More advance (not nessesary)
### Activate jupyter notebook execution in the interactive session on gpu
1. Outside of the cluster do: vim ~/.ssh/config and insert:
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
2. Start the interactive session in the cluster with:
```bash
srun --gpus=gpu:1 --mem-per-cpu=4G --time=02:00:00 --pty bash -i 
```
Wait until resources are allocated and you are in the interactive session.
3. Copy name of the node from the terminal, 
```bash
echo $SLURM_JOB_NODELIST
```
e.g. in my case it is `eu-lo-s4-065`.
4. In vs-code type `ctrl+shift+p` and type `Remote-SSH: Connect to Host...` and paste the node name.

5. Now a new vscode window whould open. Copy your eth password, becuase you may need it a couple of times now.
Open Folder, e.g. the PMLR folder. Then open the jupyter notebook, selcet your kernel and run the cells.
Test in a cell if you can access the gpu with:
```python
import torch
torch.cuda.is_available()
```
or 
```python
!nvidia-smi
``` 

## Troubleshooting
### Internet access in the interactive session
Do in the interactive session:
```bash
module load eth_proxy
```

