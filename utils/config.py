import os

import torch

# Path Configuration
OUTPUT_PATH = os.path.join("output")

LOCAL = False
if LOCAL:
    DATA_FILE_PATH = "ccai_demo/data/FCN_ERA5_data_v0/out_of_sample/"
    DATA_PATH = ""
    YEARS = [2018]
    VAL_FILE = os.path.join(DATA_PATH, "ccai_demo/data/FCN_ERA5_data_v0/out_of_sample/2018.h5")
else:
    YEARS = [2009] # , 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017
    DATA_PATH = "/cluster/scratch/mmagnus/data"
    DATA_FILE_PATH = DATA_PATH
    VAL_FILE = os.path.join(DATA_PATH, "2018_small.h5")


GLOBAL_MEANS_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/global_means.npy")
GLOBAL_STDS_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/global_stds.npy")
TIME_MEANS_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/time_means.npy")
LAND_SEA_MASK_PATH = os.path.join(DATA_PATH, "ccai_demo/additional/stats_v0/land_sea_mask.npy")

OUTPUT_FILE = 'inference_results.json'

# Training Configuration
BATCH_SIZE = 1
BATCH_SIZE_VAL = 1

EPOCHS = 10
SUBSET_TRAIN = None  # None to use all
SUBSET_VAL = None  # None to use all
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 1  # 1 will also load target so 1 -> 2
SEQUENCE_LENGTH_VAL = 1  # this is our prediction horizon
PREDICTION_LENGTH = 39  # this is our prediction horizon

# Data Dimensions and Variables
HEIGHT = 721
WIDTH = 1440
VARIABLES = [
    'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000',
    'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50', 'r500',
    'r850', 'tcwv'
]
N_VAR = 20
# print(f"Number of variables: {N_VAR}")

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


NUM_CPUS = os.cpu_count()
print(f"Number of CPUs: {NUM_CPUS}")
# num gpu
NUM_GPUS = torch.cuda.device_count()
