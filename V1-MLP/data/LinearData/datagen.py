import torch
import torch.nn as nn
import torch.optim as optim
import random



base_func = lambda x: 3*x + 1

data_input, data_output = [], []
for i in range(10):
    x = torch.tensor([[random.uniform(0, 10)]], dtype=torch.float32)   # shape: (1, 1)
    y = base_func(x)
    data_input.append(x)
    data_output.append(y)

data_input  = torch.cat(data_input, dim=0)    # (10, 1)
data_output = torch.cat(data_output, dim=0)   # (10, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = nn.Sequential(
    nn.Linear(1, 1)
).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    y_pred = model(data_input.to(device))
    loss = loss_fn(y_pred, data_output.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(20000):
    loss = train()
    print(f"Epoch {epoch:2d}  loss={loss:.4f}")

torch.save(model, "linear_model.pth")