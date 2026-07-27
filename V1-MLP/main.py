import torch
import torch.nn as nn
import torch.optim as optim

n_in = 784
n_h = 128
n_out = 10
batch_size = 32

model = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out)
)
