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
