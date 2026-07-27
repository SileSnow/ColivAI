import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 超参数
BATCH_SIZE, EPOCHS, LR, HIDDEN = 64, 10, 1e-3, 256

# 数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), BATCH_SIZE, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Sequential(nn.Flatten(), nn.Linear(784, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 训练
for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += x.size(0)
    print(f"Epoch {epoch:2d}  test_acc={correct/total:.2%}")
