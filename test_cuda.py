print("Hello world")
# print cuda available
import torch
print(f"cuda available: {torch.cuda.is_available()}")
# print device name and gpu information
if torch.cuda.is_available():
    print(f"cuda device count: {torch.cuda.device_count()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    print(f"gpu information: {torch.cuda.get_device_properties(0)}")