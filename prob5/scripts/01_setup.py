# This is notebook step 1. Run the steps in numeric order in one Python session, or run run_image_classifier.py for a standalone program.

from pathlib import Path
import platform
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.1
DATA_DIR = Path("prob5/data")
CURVES_PATH = Path("prob5/training_curves.png")
CHECKPOINT_PATH = Path("prob5/image_classifier.pt")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Python: {platform.python_version()}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
