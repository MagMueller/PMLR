
import torch
import time
import os

now = time.strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_PATH = os.path.join("output", now)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
DATA_PATH = "./ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
DATA_FILE = os.path.join(DATA_PATH, "2018.h5")
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 3
N_LAYER = 3
N_HIDDEN = 64
DT = 1.
ALPHA = 1.
GAMMA = 1.
DROPOUT = 0.1
HEIGHT = 721
WIDTH = 1440
VARIABLES = ['u10',
             'v10',
             't2m',
             'sp',
             'msl',
             't850',
             'u1000',
             'v1000',
             'z1000',
             'u850',
             'v850',
             'z850',
             'u500',
             'v500',
             'z500',
             't500',
             'z50',
             'r500',
             'r850',
             'tcwv']
N_VAR = 21  # somehow the len here is just 20
print(f"Number of variables: {N_VAR}")

if torch.backends.mps.is_available():
    DEVICE = "mps"
    # set env var PYTORCH_ENABLE_MPS_FALLBACK=1
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
