import torch
import torch.nn as nn
import torch.optim as optim

n_in = 
n_h =
n_out =
batch_size =

model = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out)
)
