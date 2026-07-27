"""
MNIST 手写数字识别 — 简单 MLP 示例
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


# ═══════════════════════════════════════
# 1. 超参数
# ═══════════════════════════════════════
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
HIDDEN_DIM = 256


# ═══════════════════════════════════════
# 2. 数据加载
# ═══════════════════════════════════════
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ═══════════════════════════════════════
# 3. 模型
# ═══════════════════════════════════════
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=HIDDEN_DIM, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: [batch, 1, 28, 28] → [batch, 784]
        x = x.view(x.size(0), -1)
        return self.net(x)


# ═══════════════════════════════════════
# 4. 训练
# ═══════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)

print(f"设备: {device}")
print(f"参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"训练集: {len(train_dataset)}  测试集: {len(test_dataset)}\n")

for epoch in range(1, EPOCHS + 1):
    # --- 训练 ---
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        correct += logits.argmax(dim=1).eq(y).sum().item()
        total += x.size(0)

    train_acc = correct / total

    # --- 测试 ---
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_correct += logits.argmax(dim=1).eq(y).sum().item()
            test_total += x.size(0)

    test_acc = test_correct / test_total
    print(f"Epoch {epoch:2d}  loss={train_loss/total:.4f}  train_acc={train_acc:.2%}  test_acc={test_acc:.2%}")


# ═══════════════════════════════════════
# 5. 预测示例
# ═══════════════════════════════════════
model.eval()
x_sample, y_sample = next(iter(test_loader))
x_sample, y_sample = x_sample[:5].to(device), y_sample[:5].to(device)
with torch.no_grad():
    preds = model(x_sample).argmax(dim=1)

print(f"\n预测示例:")
for i in range(5):
    print(f"  真实: {y_sample[i].item()}  →  预测: {preds[i].item()}  {'✅' if y_sample[i]==preds[i] else '❌'}")
