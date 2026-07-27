"""
HippocampalHub — 海马体模块
├── DentateGyrus (DG)    — 齿状回（海马体内 germinal niche）
├── EpisodicBuffer       — 情景缓存
├── detect()             — 新颖性检测（预测难度 + 模式聚类）
└── consolidate()        — 巩固蒸馏（重放→皮质柱）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from collections import deque
import random


class DentateGyrus(nn.Module):
    """
    齿状回——海马体内的 germinal niche
    
    成年大脑中唯一持续产生新神经元的区域。
    负责：模式分离 (Pattern Separation)
    ——把相似的输入编码为差异较大的输出
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 稀疏投影：d_model → d_model，鼓励模式分离
        self.projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """模式分离：使相似输入产生差异化表示"""
        # 加稀疏化噪声 + 非线性
        h = self.projection(x)
        # Top-k 稀疏化（模拟齿状回的稀疏编码特性）
        k = max(1, h.shape[-1] // 4)  # 保留 25%
        topk_vals, topk_idx = torch.topk(h.abs(), k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        h = h * mask
        return self.norm(F.gelu(h))


class EpisodicBuffer:
    """
    情景缓存——环形缓冲区
    
    存储 (input, hidden_state, loss) 三元组
    巩固阶段从中采样 → 向皮质柱蒸馏
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, x: torch.Tensor, h: torch.Tensor, loss_val: float):
        """存储一条情景记忆"""
        self.buffer.append({
            'x': x.detach().cpu(),
            'h': h.detach().cpu(),
            'loss': loss_val,
        })

    def sample(self, batch_size: int = 32) -> List[dict]:
        """随机采样一批情景记忆"""
        if len(self.buffer) == 0:
            return []
        n = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def get_high_loss_samples(self, n: int = 16) -> List[dict]:
        """获取高 loss 的样本（预测困难的样本）"""
        if len(self.buffer) == 0:
            return []
        sorted_items = sorted(self.buffer, key=lambda x: x['loss'], reverse=True)
        return sorted_items[:min(n, len(sorted_items))]

    def __len__(self):
        return len(self.buffer)


class NoveltyDetector(nn.Module):
    """
    新颖性/难度检测器
    
    不依赖"输入是否见过"（算术中所有组合都可能见过）
    而是检测"预测难度 + 梯度冲突"
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 轻量自编码器：输入 → 瓶颈 → 重构
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )
        self.decoder = nn.Linear(d_model // 2, d_model)

        # 滑动平均：跟踪近期平均重构误差
        self.register_buffer("avg_recon_error", torch.tensor(0.0))
        self.register_buffer("std_recon_error", torch.tensor(1.0))
        self.momentum = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Returns:
            encoded: 编码后的表示
            novelty_score: 新颖性分数 ∈ [0, 1]
        """
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        recon_error = F.mse_loss(recon, x.detach(), reduction='none').mean(dim=-1)

        # 更新滑动统计
        with torch.no_grad():
            batch_avg = recon_error.mean()
            batch_std = recon_error.std()
            self.avg_recon_error = (1 - self.momentum) * self.avg_recon_error + self.momentum * batch_avg
            self.std_recon_error = (1 - self.momentum) * self.std_recon_error + self.momentum * (batch_std + 1e-8)

        # Z-score → sigmoid → [0, 1]
        z_score = (recon_error.mean() - self.avg_recon_error) / (self.std_recon_error + 1e-8)
        novelty_score = torch.sigmoid(z_score).item()

        return encoded, novelty_score


class HippocampalHub(nn.Module):
    """
    海马体中枢
    
    三大角色：
    1. 新颖性/难度检测 → 给 SparkTrigger 提供信号
    2. 情景缓存 → 存储重要样本
    3. 巩固蒸馏 → 向皮质柱传递知识
    """

    def __init__(self, d_model: int, buffer_size: int = 256, distill_lr: float = 1e-3):
        super().__init__()
        self.d_model = d_model

        # 齿状回（模式分离）
        self.dentate_gyrus = DentateGyrus(d_model)

        # 新颖性检测器
        self.novelty_detector = NoveltyDetector(d_model)

        # 情景缓存
        self.buffer = EpisodicBuffer(buffer_size)

        # 蒸馏投影（海马体表示 → 皮质柱表示）
        self.distill_proj = nn.Linear(d_model, d_model)

        # 统计
        self.register_buffer("total_seen", torch.zeros(1))
        self.register_buffer("total_novel", torch.zeros(1))

    def forward(self, x: torch.Tensor, task_loss: float) -> Tuple[torch.Tensor, float, bool]:
        """
        处理输入，返回编码 + 新颖性分数 + 是否应该缓存

        Args:
            x: [batch, seq, d_model] 输入
            task_loss: 当前 batch 的任务损失

        Returns:
            encoded: 编码后的海马体表示
            novelty_score: 新颖性分数
            should_cache: 是否建议存入情景缓存
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        # 齿状回模式分离
        dg_out = self.dentate_gyrus(x_flat)

        # 新颖性检测
        encoded, novelty_score = self.novelty_detector(dg_out)
        encoded = encoded.reshape(B, T, -1)

        # 判断是否缓存：高 loss 或高新颖性
        should_cache = (novelty_score > 0.5) or (task_loss > 1.0)

        self.total_seen += 1
        if should_cache:
            self.total_novel += 1
            self.buffer.store(x.detach(), dg_out.detach(), task_loss)

        return encoded, novelty_score, should_cache

    def consolidate(self, columns: List, corpus_gate, optimizer,
                    batch_size: int = 16, epochs: int = 3) -> float:
        """
        巩固蒸馏：从情景缓存采样 → 蒸馏到皮质柱

        Returns:
            平均蒸馏损失
        """
        if len(self.buffer) == 0:
            return 0.0

        total_distill_loss = 0.0
        n_batches = 0

        for _ in range(epochs):
            samples = self.buffer.get_high_loss_samples(batch_size)
            if not samples:
                break

            for sample in samples:
                x = sample['x'].to(next(self.parameters()).device)
                h_target = sample['h'].to(next(self.parameters()).device)
                h_target = h_target.reshape(x.shape[0], x.shape[1], -1)

                # 在海马体内部做蒸馏
                dg_out = self.dentate_gyrus(x.reshape(-1, self.d_model))
                h_pred = self.distill_proj(dg_out)
                h_pred = h_pred.reshape(x.shape[0], x.shape[1], -1)

                distill_loss = F.mse_loss(h_pred, h_target)

                optimizer.zero_grad()
                distill_loss.backward()
                optimizer.step()

                total_distill_loss += distill_loss.item()
                n_batches += 1

        return total_distill_loss / (n_batches + 1)

    def get_novelty_rate(self) -> float:
        """新颖性率：被标记为新颖的输入比例"""
        if self.total_seen == 0:
            return 0.0
        return (self.total_novel / self.total_seen).item()


# ═══════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    d_model = 64
    hippo = HippocampalHub(d_model)

    x = torch.randn(2, 10, d_model)
    encoded, novelty, cache = hippo(x, task_loss=0.5)

    print(f"编码: {encoded.shape}")
    print(f"新颖性分数: {novelty:.4f}, 缓存?: {cache}")
    print(f"缓存大小: {len(hippo.buffer)}")
    print(f"新颖性率: {hippo.get_novelty_rate():.2%}")
