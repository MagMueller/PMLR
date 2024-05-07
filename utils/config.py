import os
import time
import torch

# Path Configuration
now = time.strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_PATH = os.path.join("output", now)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

DATA_PATH = "./ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
DATA_PATH = "/cluster/scratch/mmagnus/data"

YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
# for local testing
# YEARS = [2018, 2018]

VAL_FILE = os.path.join(DATA_PATH, "2017.h5")

GLOBAL_MEANS_PATH = "./ccai_demo/additional/stats_v0/global_means.npy"
GLOBAL_STDS_PATH = "./ccai_demo/additional/stats_v0/global_stds.npy"

TIME_MEANS_PATH = "./ccai_demo/additional/stats_v0/time_means.npy"
LAND_SEA_MASK_PATH = "./ccai_demo/additional/stats_v0/land_sea_mask.npy"

# Training Configuration
BATCH_SIZE = 1
EPOCHS = 2
SUBSET_TRAIN = 1  # None to use all
SUBSET_VAL = 1  # None to use all
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 1

# Model Architecture
N_LAYER = 3
N_HIDDEN = 64
DT = 1.
ALPHA = 1.
GAMMA = 1.
DROPOUT = 0.1

# Data Dimensions and Variables
HEIGHT = 721
WIDTH = 1440
VARIABLES = [
    'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000',
    'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50', 'r500',
    'r850', 'tcwv'
]
N_VAR = 21
print(f"Number of variables: {N_VAR}")

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"