"""
verify3.1 — Token Embedding 预训练
目标：让模型在开始算术训练前，就知道数字和运算符是两类东西。

训练目标：
1. 数字聚簇 (digit-digit similarity ≈ 0.3)
2. 运算符聚簇 (op-op similarity ≈ 0.5)  
3. 数字与运算符正交 (digit-op similarity ≈ 0)
4. 数轴排序: cos(n, n+1) > cos(n, n+3)，数值越近越相似
5. 防坍缩: 每个数字仍可区分 (同数字 identity=1.0)

输出：embedding.pth (16×128 的权重矩阵)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 配置
# ============================================================
VOCAB_SIZE = 16
D_MODEL = 128
N_EPOCHS = 2000
LR = 5e-3
SEED = 42

DIGIT_IDX = list(range(10))     # 0-9
OP_IDX = [10, 11]               # +, -
SPECIAL_IDX = [12, 13, 14, 15]  # =, PAD, BOS, EOS

TOKEN_NAMES = ['0','1','2','3','4','5','6','7','8','9','+','-','=','PAD','BOS','EOS']

torch.manual_seed(SEED)

# ============================================================
# 模型: 一个可学习的 embedding 矩阵
# ============================================================
class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB_SIZE, D_MODEL) * 0.1)
    
    def forward(self, x):
        return F.embedding(x, self.weight)

model = TokenEmbedding()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

# ============================================================
# 损失函数
# ============================================================
def compute_loss(emb):
    """
    emb: [16, 128] 原始 embedding 权重
    """
    # L2 归一化
    emb_norm = F.normalize(emb, dim=1)  # [16, 128]
    
    digits = emb_norm[DIGIT_IDX]   # [10, 128]
    ops = emb_norm[OP_IDX]         # [2, 128]
    specials = emb_norm[SPECIAL_IDX]  # [4, 128]
    
    loss = 0.0
    
    # ---- 1. 数字聚簇 ----
    digit_sim = digits @ digits.T  # [10, 10]
    # 对角线 = 1.0 (自己和自己)，非对角线目标 = 0.35
    target_digit = torch.eye(10, device=emb.device) * 1.0 + \
                   (1 - torch.eye(10, device=emb.device)) * 0.35
    loss += F.mse_loss(digit_sim, target_digit) * 10.0  # 权重高，这是核心目标
    
    # ---- 2. 运算符聚簇 ----
    op_sim = ops @ ops.T  # [2, 2]
    target_op = torch.eye(2, device=emb.device) * 1.0 + \
                (1 - torch.eye(2, device=emb.device)) * 0.5
    loss += F.mse_loss(op_sim, target_op) * 5.0
    
    # ---- 3. 数字与运算符正交 ----
    cross_sim = digits @ ops.T  # [10, 2]
    loss += (cross_sim ** 2).mean() * 20.0  # 推往 0
    
    # ---- 4. 特殊 token 与数字/运算符分离 ----
    digit_special = digits @ specials.T  # [10, 4]
    op_special = ops @ specials.T        # [2, 4]
    loss += (digit_special ** 2).mean() * 2.0
    loss += (op_special ** 2).mean() * 2.0
    
    # ---- 5. 数轴排序：相邻数字比远离数字更相似 ----
    margin_loss = 0.0
    n_pairs = 0
    for anchor in range(10):
        for pos in range(10):
            if pos == anchor:
                continue
            dist = abs(anchor - pos)
            for neg in range(10):
                if neg == anchor or abs(anchor - neg) <= dist:
                    continue
                # pos 比 neg 更近 → sim(anchor, pos) > sim(anchor, neg)
                sim_pos = digits[anchor] @ digits[pos]
                sim_neg = digits[anchor] @ digits[neg]
                margin_loss += torch.relu(sim_neg - sim_pos + 0.05)
                n_pairs += 1
    if n_pairs > 0:
        loss += margin_loss / n_pairs * 2.0
    
    # ---- 6. 防坍缩：方差不能太小 ----
    digit_std = digits.std(dim=0).mean()
    loss += torch.relu(0.05 - digit_std) * 5.0
    
    return loss

# ============================================================
# 训练
# ============================================================
losses = []
print(f"开始训练 token embedding ({N_EPOCHS} epochs)...")
print(f"{'Epoch':>6s} | {'Loss':>10s} | {'d-d sim':>9s} | {'d-op sim':>9s} | {'op-op':>7s} | {'adj-far':>8s}")
print("-" * 65)

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    loss = compute_loss(model.weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    
    if epoch % 200 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            emb_norm = F.normalize(model.weight, dim=1)
            digits = emb_norm[DIGIT_IDX]
            ops = emb_norm[OP_IDX]
            
            d_d = (digits @ digits.T).mean().item()
            d_op = (digits @ ops.T).abs().mean().item()
            o_o = (ops @ ops.T)[0,1].item()
            
            # 相邻相似度 vs 间隔5相似度
            adj = sum((digits[i] @ digits[i+1]).item() for i in range(9)) / 9
            far = sum((digits[i] @ digits[i+5]).item() for i in range(5)) / 5
            
        print(f"{epoch:>6d} | {loss.item():>10.4f} | {d_d:>9.4f} | {d_op:>9.4f} | {o_o:>7.4f} | {adj:.4f}>{far:.4f}")

print("\n✅ 训练完成!")

# ============================================================
# 最终分析
# ============================================================
with torch.no_grad():
    emb_norm = F.normalize(model.weight, dim=1)
    sim = emb_norm @ emb_norm.T
    
    # 打印相似度矩阵
    print("\n==================== 最终相似度矩阵 ====================")
    header = "      " + "".join(f"{TOKEN_NAMES[i]:>6s}" for i in range(16))
    print(header)
    for i in range(16):
        row = f"{TOKEN_NAMES[i]:>4s} |"
        for j in range(16):
            row += f"{sim[i,j].item():>6.2f}"
        print(row)
    
    # 关键指标
    print("\n==================== 关键指标 ====================")
    
    # 数字间相似度
    digit_pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            digit_pairs.append(sim[i,j].item())
    print(f"数字-数字 平均相似度: {np.mean(digit_pairs):.4f} (目标 ~0.35)")
    
    # 数字-运算符相似度
    d_op_pairs = []
    for i in range(10):
        for j in [10, 11]:
            d_op_pairs.append(abs(sim[i,j].item()))
    print(f"数字-运算符 平均|相似度|: {np.mean(d_op_pairs):.4f} (目标 ~0)")
    
    # 运算符间相似度
    print(f"+ 与 - 相似度: {sim[10,11].item():.4f} (目标 ~0.5)")
    
    # 数轴排序检查
    adj_preserved = 0
    total_checks = 0
    for i in range(10):
        for d1 in range(1, 5):
            for d2 in range(d1+1, min(6, 10-i)):
                if i + d1 < 10 and i + d2 < 10:
                    if sim[i, i+d1] > sim[i, i+d2]:
                        adj_preserved += 1
                    total_checks += 1
    print(f"数轴排序正确率: {adj_preserved/total_checks*100:.1f}% ({adj_preserved}/{total_checks})")
    
    # 混淆检查
    bad = 0
    for i in range(10):
        min_digit = min(sim[i,j].item() for j in range(10) if j != i)
        max_op = max(sim[i,j].item() for j in [10, 11])
        if max_op > min_digit:
            bad += 1
    print(f"数字-运算符混淆: {bad}/10 个数字 (应该为 0!)")

# ============================================================
# 保存
# ============================================================
save_path = '/root/airesearch/verify3.1/embedding.pth'
torch.save(model.weight.data.clone(), save_path)
print(f"\n💾 保存到 {save_path}")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Loss 曲线
ax = axes[0]
ax.plot(losses, alpha=0.5, linewidth=0.5)
ax.plot(np.convolve(losses, np.ones(50)/50, mode='valid'), linewidth=2, color='red')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Loss Curve')

# 2. 相似度矩阵热力图
ax = axes[1]
im = ax.imshow(sim.numpy(), cmap='RdBu_r', vmin=-0.3, vmax=1.0)
ax.set_xticks(range(16)); ax.set_yticks(range(16))
ax.set_xticklabels(TOKEN_NAMES, fontsize=8)
ax.set_yticklabels(TOKEN_NAMES, fontsize=8)
ax.set_title('Cosine Similarity Matrix')
plt.colorbar(im, ax=ax, shrink=0.8)

# 3. 数字相似度随距离衰减
ax = axes[2]
distances = range(1, 10)
means = []
for d in distances:
    vals = [sim[i, i+d].item() for i in range(10-d)]
    means.append(np.mean(vals))
ax.bar(distances, means)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_xlabel('Numerical Distance'); ax.set_ylabel('Avg Similarity')
ax.set_title('Digit Similarity vs Distance')

plt.tight_layout()
fig_path = '/root/airesearch/verify3.1/embedding_analysis.png'
plt.savefig(fig_path, dpi=150)
print(f"📊 分析图保存到 {fig_path}")
