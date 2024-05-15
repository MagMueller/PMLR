from utils.dotdict import DotDict
import os
import torch


def get_config():

    # Configuration initialization
    config = DotDict()

    # Path Configuration
    config.OUTPUT_PATH = os.path.join("output")
    config.LOCAL = True

    if config.LOCAL:
        config.DATA_FILE_PATH = "ccai_demo/data/FCN_ERA5_data_v0/out_of_sample/"
        if not os.path.exists(config.DATA_FILE_PATH):
            raise FileNotFoundError(
                f"Data path {config.DATA_FILE_PATH} does not exist \n please do the following: \n wget https://portal.nersc.gov/project/m4134/ccai_demo.tar \n tar -xvf ccai_demo.tar \n rm ccai_demo.tar")
        config.DATA_PATH = ""
        config.YEARS = [2018]
        config.VAL_FILE = os.path.join(config.DATA_PATH, "ccai_demo/data/FCN_ERA5_data_v0/out_of_sample/2018.h5")
        config.STRATEGY = "auto"
    else:
        config.YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
        config.DATA_PATH = "/cluster/scratch/mmagnus/data"
        config.DATA_FILE_PATH = config.DATA_PATH
        config.VAL_FILE = os.path.join(config.DATA_PATH, "2018.h5")
        config.STRATEGY = "ddp"

    config.GLOBAL_MEANS_PATH = os.path.join(config.DATA_PATH, "ccai_demo/additional/stats_v0/global_means.npy")
    config.GLOBAL_STDS_PATH = os.path.join(config.DATA_PATH, "ccai_demo/additional/stats_v0/global_stds.npy")
    config.TIME_MEANS_PATH = os.path.join(config.DATA_PATH, "ccai_demo/additional/stats_v0/time_means.npy")
    config.LAND_SEA_MASK_PATH = os.path.join(config.DATA_PATH, "ccai_demo/additional/stats_v0/land_sea_mask.npy")

    config.OUTPUT_FILE = 'inference_results.json'

    # Training Configuration
    config.BATCH_SIZE = 1
    config.BATCH_SIZE_VAL = 1

    config.EPOCHS = 20
    config.SUBSET_TRAIN = None
    config.SUBSET_VAL = None
    config.LEARNING_RATE = 0.001
    config.SEQUENCE_LENGTH = 1
    config.SEQUENCE_LENGTH_VAL = 1
    config.PREDICTION_LENGTH = 39

    # Data Dimensions and Variables
    config.HEIGHT = 721
    config.WIDTH = 1440

    config.VARIABLES = [
        'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000',
        'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50', 'r500',
        'r850', 'tcwv'
    ]
    config.N_VAR = 20

    # model
    config.MODEL_CONFIG = {
        "num_layers": 3,
        "hidden_dim": 64,
        "dropout": 0.1,
        "num_heads": 8,
        "input_dim": config.N_VAR,
        "output_dim": config.N_VAR,
        "time_dim": 1
    }

    # Compute-dependent configuration
    config.DEVICE = "cpu"
    if torch.cuda.is_available():
        config.DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        config.DEVICE = "mps"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    config.NUM_CPUS = os.cpu_count()
    config.CPU_MEM = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)
    if torch.cuda.is_available():
        config.NUM_GPUS = torch.cuda.device_count()
        config.GPU_MEM = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(config.NUM_GPUS)]

    # Print out current configuration
    print(config)

    return config
