import torch
import transformers
import drain3
import sklearn
import pandas

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("transformers:", transformers.__version__)
print("all imports OK")