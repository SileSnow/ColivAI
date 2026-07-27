"""
CorpusGate — 胼胝体门控（MoE 路由）
HebbianBias — 赫布共激活偏置矩阵

负责：将 token 路由到 top-k 皮质柱，输出加权组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class HebbianBias(nn.Module):
    """
    赫布共激活偏置矩阵
    "Columns that fire together, wire together."

    维护一个 [num_cols, num_cols] 矩阵，
    记录专家之间的共激活频率。
    路由时加到 logits 上，鼓励常合作的专家继续合作。
    """

    def __init__(self, max_columns: int):
        super().__init__()
        self.max_columns = max_columns
        # 共激活计数（不是参数，是统计量）
        self.register_buffer("coactivation", torch.zeros(max_columns, max_columns))
        self.register_buffer("total_steps", torch.zeros(1))

    def update(self, selected_indices: torch.Tensor):
        """
        更新共激活统计
        Args:
            selected_indices: [num_tokens, top_k] 被选中的专家索引
        """
        with torch.no_grad():
            # 为每对 (i,j) 计数同时被选中的次数
            for b in range(selected_indices.shape[0]):
                idx = selected_indices[b]  # [top_k]
                for i in range(len(idx)):
                    for j in range(len(idx)):
                        if idx[i] < self.max_columns and idx[j] < self.max_columns:
                            self.coactivation[idx[i], idx[j]] += 1
            self.total_steps += 1

    def get_bias(self, num_active: int) -> torch.Tensor:
        """获取归一化的共激活偏置 [num_active, num_active]"""
        if self.total_steps == 0 or num_active == 0:
            return torch.zeros(num_active, num_active)
        # 只取活跃的柱
        bias = self.coactivation[:num_active, :num_active] / (self.total_steps + 1)
        return bias * 0.1  # 缩放到小幅度


class CorpusGate(nn.Module):
    """
    胼胝体门控——MoE 路由中枢

    职责：
    1. 对每个 token 计算路由分数
    2. Top-k 选择激活的柱
    3. 加权组合柱的输出
    4. 计算负载均衡损失
    """

    def __init__(self, d_model: int, max_columns: int, top_k: int = 2,
                 noise_std: float = 0.1, load_balance_alpha: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.max_columns = max_columns
        self.top_k = top_k
        self.noise_std = noise_std
        self.load_balance_alpha = load_balance_alpha

        # 路由器：线性投影 d_model → num_columns
        self.router = nn.Linear(d_model, max_columns, bias=False)

        # 赫布偏置
        self.hebbian = HebbianBias(max_columns)

        # 统计
        self.register_buffer("expert_counts", torch.zeros(max_columns))

    def forward(self, x: torch.Tensor, column_outputs: List[torch.Tensor],
                column_feedbacks: List[torch.Tensor],
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, d_model] 原始嵌入（用于计算路由分数）
            column_outputs: List of [batch, seq, d_model] 每个柱的输出
            column_feedbacks: List of [batch, seq, d_model] 每个柱的反馈
            training: 是否训练模式（影响噪声门控）

        Returns:
            output: [batch, seq, d_model] 加权组合输出
            load_balance_loss: 负载均衡损失
            selected_indices: [batch*seq, top_k] 被选中的柱索引
        """
        B, T, D = x.shape
        num_active = len(column_outputs)
        device = x.device

        # 1. 计算路由 logits（只取活跃柱数）
        router_logits = self.router(x)[:, :, :num_active]  # [B, T, num_active]

        # 2. 噪声门控（训练时）
        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # 3. 加赫布偏置（[num_active, num_active] → 对每个柱的偏置）
        hebb_bias = self.hebbian.get_bias(num_active).to(device)  # [num_active, num_active]
        column_bias = hebb_bias.mean(dim=-1)  # [num_active] — 每个柱的平均共激活度
        router_logits = router_logits + column_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, num_active]

        # 4. 加反馈调制（L6 feedback 对门控的影响）
        if column_feedbacks:
            # feedback: 每个柱有一个 [B,T,D] 的反馈信号
            # 用 feedback 的 norm 调制该柱的路由分数
            fb_scores = torch.stack([
                fb.mean(dim=-1) for fb in column_feedbacks
            ], dim=-1)  # [num_active, B, T] → [B, T, num_active]
            router_logits = router_logits + fb_scores * 0.01

        # 5. Softmax → Top-k
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, num_active]

        # Top-k 选择（确保 k ≤ num_active）
        k = min(self.top_k, num_active)
        topk_probs, topk_indices = torch.topk(router_probs, k, dim=-1)
        # topk_probs: [B, T, top_k], topk_indices: [B, T, top_k]

        # 归一化 top-k 概率（除以被选中的概率之和）
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 6. 加权组合输出
        output = torch.zeros_like(x)
        for ki in range(k):
            idx = topk_indices[:, :, ki]  # [B, T]
            prob = topk_probs[:, :, ki]   # [B, T]
            for c in range(num_active):
                mask_c = (idx == c)
                if mask_c.any():
                    output[mask_c] += prob[mask_c].unsqueeze(-1) * column_outputs[c][mask_c]

        # 7. 负载均衡损失
        flat_indices = topk_indices.reshape(-1, k)  # [B*T, k]
        expert_counts = torch.zeros(num_active, device=device)
        expert_probs = router_probs.mean(dim=(0, 1))  # [num_active]

        for c in range(num_active):
            expert_counts[c] = (flat_indices == c).sum().float()

        f_c = expert_counts / (B * T * k + 1e-8)

        # Switch Transformer 风格的负载均衡损失
        load_balance_loss = (num_active * (f_c * expert_probs).sum())

        # 更新统计
        if training:
            self.hebbian.update(flat_indices)
        with torch.no_grad():
            self.expert_counts[:num_active] += expert_counts

        return output, load_balance_loss, topk_indices

    def get_expert_utilization(self, num_active: int) -> torch.Tensor:
        """返回每个柱的利用率分布（用于 SparkTrigger）"""
        total = self.expert_counts[:num_active].sum()
        if total == 0:
            return torch.ones(num_active) / num_active
        return self.expert_counts[:num_active] / total


# ═══════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    d_model, max_cols, top_k = 64, 5, 2
    gate = CorpusGate(d_model, max_cols, top_k)

    # 模拟 3 个柱的输出
    x = torch.randn(2, 10, d_model)
    col_outputs = [torch.randn(2, 10, d_model) for _ in range(3)]
    col_feedbacks = [torch.randn(2, 10, d_model) for _ in range(3)]

    out, lb_loss, indices = gate(x, col_outputs, col_feedbacks)
    print(f"输出: {out.shape}, 负载均衡损失: {lb_loss.item():.4f}")
    print(f"利用率: {gate.get_expert_utilization(3)}")
