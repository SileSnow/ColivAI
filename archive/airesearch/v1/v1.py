# 导入库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 初始化模型基本参数
n_in = 3          # 输入层大小：[操作数1, 操作符, 操作数2]
n_h = 16          # 隐藏层大小
n_out = 1         # 输出层大小：计算结果
batch_size = 16   # 每批样本数
epoches = 200    # 训练轮数

# -------------------- 加载数据 --------------------
print("加载数据...")
data = torch.load("data.pt")
X = torch.tensor(data["X"], dtype=torch.float32)
Y = torch.tensor(data["Y"], dtype=torch.float32).reshape(-1, 1)

print(f"X 形状: {X.shape}")
print(f"Y 形状: {Y.shape}")

# 封装成 DataLoader
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------- 搭建模型 --------------------
model = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out)
)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    threshold=1e-4,
    cooldown=0,
    min_lr=0.0,
)

# -------------------- 训练 --------------------
def train():
    model.train()
    for epoch in range(epoches):
        total_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epoches}], Loss: {avg_loss:.6f}")
    
    # 保存完整模型（架构 + 参数）
    torch.save(model, "model.pth")
    print("✅ 模型已保存到 model.pth")

# -------------------- 测试 --------------------
def test():
    model.eval()
    with torch.no_grad():
        samples = torch.tensor([
            [23, 0, 45],   # 23 + 45 = 68
            [50, 1, 12],   # 50 - 12 = 38
            [7,  0, 8],    # 7  + 8  = 15
            [99, 1, 55],   # 99 - 55 = 44
        ], dtype=torch.float32)
        predictions = model(samples)
        for i, (a, op, b) in enumerate(samples.int()):
            op_str = "+" if op == 0 else "-"
            print(f"  {a.item()} {op_str} {b.item()} = {predictions[i].item():.1f}")

# -------------------- 开跑 --------------------
if __name__ == "__main__":
    train()
    print("\n训练完成！测试一下：")
    test()
