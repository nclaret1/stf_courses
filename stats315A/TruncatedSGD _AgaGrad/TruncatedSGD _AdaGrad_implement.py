file_path = os.path.abspath("/Users/noemieclaret/Desktop/")
sys.path.append(file_path)
from beyond_gd_starter import *;
from mnist_experiments import FitNN
from mnist_experiments import FitNN

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ttest_ind



experiments = [
    ("mlp", "adam"),
    ("mlp", "adagrad"),
    ("mlp", "truncatedsgd"),
    ("mlp", "truncatedadagrad"),
    ("conv", "adam"),
    ("conv", "adagrad"),
    ("conv", "truncatedsgd"),
    ("conv", "truncatedadagrad"),
]

for model_type, optimizer in experiments:
    print(f"\nRunning experiment: Model={model_type.upper()}, Optimizer={optimizer.upper()}...\n")
    FitNN(model_type=model_type, optimizer_type=optimizer, learning_rate=1e-2)
