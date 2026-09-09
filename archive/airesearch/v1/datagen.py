# 生成一百以内加减法数据集（操作符编码：0=加法，1=减法）
import torch

X = []
Y = []

for op in [0, 50]:
    for a in range(100):
        for b in range(100):
            X.append([a, op, b])
            Y.append(a + b if op == 0 else a - b)

# 看一眼前 5 个样本
for i in range(5):
    a, op, b = torch.tensor(X[i]).int().tolist()
    op_str = "+" if op == 0 else "-"
    print(f"  样本 {i}: [{a}, {op}, {b}] → {a} {op_str} {b} = {torch.tensor(Y[i]).item():.0f}")

# 保存到文件
torch.save({"X": X, "Y": Y}, "data.pt")
print("✅ 已保存到 data.pt")
