import os

import torch

# Path Configuration
OUTPUT_PATH = os.path.join("output")

LOCAL = True
if LOCAL:
    DATA_FILE_PATH = "ccai_demo/data/FCN_ERA5_data_v0/out_of_sample/"
    # check if exists
    if not os.path.exists(DATA_FILE_PATH):
        # print download from
        # wget https://portal.nersc.gov/project/m4134/ccai_demo.tar
        # tar -xvf ccai_demo.tar
        # rm ccai_demo.tar
        raise FileNotFoundError(
            f"Data path {DATA_FILE_PATH} does not exist \n please do the following: \n wget https://portal.nersc.gov/project/m4134/ccai_demo.tar \n tar -xvf ccai_demo.tar \n rm ccai_demo.tar")
    DATA_PATH = ""
    YEARS = [2018]
    VAL_FILE = os.path.join(DATA_PATH, "ccai_demo/data/FCN_ERA5_data_v0/out_of_sample/2018.h5")
    STRATEGY = "auto"

else:
    YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    DATA_PATH = "/cluster/scratch/mmagnus/data"
    DATA_FILE_PATH = DATA_PATH
    VAL_FILE = os.path.join(DATA_PATH, "2018.h5")  # 2018_small.h5
    STRATEGY = "ddp"


GLOBAL_MEANS_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/global_means.npy")
GLOBAL_STDS_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/global_stds.npy")
TIME_MEANS_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/time_means.npy")
LAND_SEA_MASK_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/land_sea_mask.npy")

OUTPUT_FILE = 'inference_results.json'

# Training Configuration
BATCH_SIZE = 1
BATCH_SIZE_VAL = 1

EPOCHS = 20
SUBSET_TRAIN = None  # None to use all
SUBSET_VAL = None  # None to use all
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 1  # 1 will also load target so 1 -> 2
SEQUENCE_LENGTH_VAL = 1  # this is our prediction horizon
PREDICTION_LENGTH = 39  # this is our prediction horizon

# Data Dimensions and Variables
X1 = 0
X2 = 10
Y1 = 0
Y2 = 10


HEIGHT = Y2 - Y1  # 721
WIDTH = X2 - X1  # 1440
VARIABLES = [
    'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000',
    'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50', 'r500',
    'r850', 'tcwv'
]
N_VAR = 20
# print(f"Number of variables: {N_VAR}")

MODEL_NAME = "deep_coRNN"  # "coRNN" or "coRNN2" or "deep_coRNN"
if MODEL_NAME == "coRNN":
    MODEL_CONFIG = {
        "n_inp": N_VAR,
        "n_hid": 64,
        "n_out": N_VAR,
        "dt": 1.,
        "epsilon": 1.,
        "gamma": 1.,
    }
elif MODEL_NAME == "coRNN2":
    MODEL_CONFIG = {
        "n_inp": N_VAR,
        "n_hid": 64,
        "n_out": N_VAR,
        "dt": 1.,
        "epsilon": 1.,
        "gamma": 1.,
    }
elif MODEL_NAME == "deep_coRNN":
    MODEL_CONFIG = {
        "nfeat": N_VAR,
        "nhid": 64,
        "nclass": N_VAR,
        "nlayers": 3,
        "dt": 1.,
        "alpha": 1.,
        "gamma": 1.,
        "dropout": 0.1
    }
else:
    raise ValueError(f"Model name {MODEL_NAME} not recognized")


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    DEVICE = "cpu"

NUM_CPUS = os.cpu_count()
print(f"Number of CPUs: {NUM_CPUS}")
# memery per cpu
CPU_MEM = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)
print(f"Memory CPU in GB: {CPU_MEM}")
# num gpu
if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    print(f"Number of GPUs: {NUM_GPUS}")
    for i in range(NUM_GPUS):
        print(f"Memory device {i} in GB: {torch.cuda.get_device_properties(i).total_memory / 1e9}")
