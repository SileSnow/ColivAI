"""
SparkColumn — 单个皮质柱（MoE 专家）
四层结构：L4_Input → L23_Assoc → L5_Output → L6_Feedback
Neuroblast — 未成熟的皮质柱（渐进生长中）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════
# 可学习相对位置编码（PE 探针）
# ═══════════════════════════════════════════════════

class LearnableRelBias(nn.Module):
    """
    rel_bias(d) = w * tanh(v * d) + sech^2(tau * d)
    每个头独立 3 参数: w, v, log_tau
    同时作为 PE 探针——监控参数变化率
    """

    def __init__(self, n_heads: int, max_len: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len

        # 非零初始化（避免梯度死锁）
        w_init = torch.randn(n_heads) * 0.1
        v_init = torch.randn(n_heads) * 0.02 + 0.01

        self.w = nn.Parameter(w_init)
        self.v = nn.Parameter(v_init)
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

        # PE 探针缓存
        self.register_buffer("_prev_w", w_init.clone())
        self.register_buffer("_prev_v", v_init.clone())
        self.register_buffer("_prev_log_tau", torch.zeros(n_heads))
        self.register_buffer("distances", None, persistent=False)

    def _get_bias_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.distances is None or self.distances.shape[0] < seq_len:
            pos = torch.arange(seq_len, device=device)
            dists = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
            self.distances = dists.to(device)

        d = self.distances[:seq_len, :seq_len]
        w = self.w.view(self.n_heads, 1, 1)
        v_abs = self.v.abs().view(self.n_heads, 1, 1)
        tau = torch.exp(self.log_tau).view(self.n_heads, 1, 1)

        tanh_part = w * torch.tanh(v_abs * d.unsqueeze(0))
        cosh_input = torch.clamp(tau * d.unsqueeze(0), max=85.0)
        sech2_part = 1.0 / (torch.cosh(cosh_input) ** 2 + 1e-8)

        return tanh_part + sech2_part  # [n_heads, seq, seq]

    def get_activity(self) -> float:
        """PE 探针：返回参数变化率 ∈ [0,1]"""
        with torch.no_grad():
            dw = (self.w - self._prev_w).abs().mean().item()
            dv = (self.v - self._prev_v).abs().mean().item()
            dt = (self.log_tau - self._prev_log_tau).abs().mean().item()
            ws = self.w.abs().mean().item() + 1e-8
            vs = self.v.abs().mean().item() + 1e-8
            ts = self.log_tau.abs().mean().item() + 1e-8
            return (dw / ws + dv / vs + dt / ts) / 3.0

    def update_prev(self):
        with torch.no_grad():
            self._prev_w.copy_(self.w)
            self._prev_v.copy_(self.v)
            self._prev_log_tau.copy_(self.log_tau)


# ═══════════════════════════════════════════════════
# 皮质柱内部层
# ═══════════════════════════════════════════════════

class L4_InputLayer(nn.Module):
    """L4 内颗粒层：丘脑输入投影"""
    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.projection(x)))


class L23_AssocLayer(nn.Module):
    """L2/3 关联层：自注意力 + 可学习相对位置偏置"""

    def __init__(self, d_model: int, n_heads: int, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.rel_bias = LearnableRelBias(n_heads, max_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        residual = x
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        rel_bias = self.rel_bias._get_bias_matrix(T, x.device)
        attn_scores = attn_scores + rel_bias.unsqueeze(0)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(attn_out)

        if residual.dim() == 2:
            out = out.squeeze(0)

        return self.norm(residual + out)

    def get_pe_activity(self) -> float:
        return self.rel_bias.get_activity()

    def update_pe_prev(self):
        self.rel_bias.update_prev()


class L5_OutputLayer(nn.Module):
    """L5 内锥体层：FFN 主输出"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(x))


class L6_FeedbackLayer(nn.Module):
    """L6 多形层：反馈到门控"""
    def __init__(self, d_model: int):
        super().__init__()
        self.feedback = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feedback(x)


# ═══════════════════════════════════════════════════
# SparkColumn — 完整皮质柱
# ═══════════════════════════════════════════════════

class SparkColumn(nn.Module):
    """
    完整皮质柱（MoE 专家）
    四层结构，支持渐进生长
    生命周期: Neuroblast(0-3层) → SparkColumn(4层成熟)
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int,
                 max_len: int = 128, num_layers: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.max_layers = 4
        self.growth_order = ["L6", "L5", "L4", "L23"]

        self.l6: Optional[L6_FeedbackLayer] = None
        self.l5: Optional[L5_OutputLayer] = None
        self.l4: Optional[L4_InputLayer] = None
        self.l23: Optional[L23_AssocLayer] = None

        for i in range(min(num_layers, self.max_layers)):
            self._add_layer(self.growth_order[i])

        self.output_norm = nn.LayerNorm(d_model)

    def _add_layer(self, name: str):
        if name == "L6":
            self.l6 = L6_FeedbackLayer(self.d_model)
        elif name == "L5":
            self.l5 = L5_OutputLayer(self.d_model, self.d_ff)
        elif name == "L4":
            self.l4 = L4_InputLayer(self.d_model)
        elif name == "L23":
            self.l23 = L23_AssocLayer(self.d_model, self.n_heads)

    def grow(self) -> bool:
        """渐进生长一步。返回 True=还在生长，False=已成熟"""
        if self.num_layers >= self.max_layers:
            return False
        name = self.growth_order[self.num_layers]
        self._add_layer(name)
        self.num_layers += 1
        if self.num_layers > 1:
            self._project_init(name)
        return self.num_layers < self.max_layers

    def _project_init(self, new_name: str):
        """从已有最深层做投影初始化 + 微噪声"""
        noise = 0.01
        new_layer = getattr(self, new_name.lower())
        # 找最深层已有层做种子
        for old_name in reversed(self.growth_order[:self.num_layers - 1]):
            old_layer = getattr(self, old_name.lower())
            if old_layer is not None:
                for (nn, np), (on, op) in zip(
                    new_layer.named_parameters(), old_layer.named_parameters()):
                    if 'weight' in nn and np.shape == op.shape:
                        np.data.copy_(op.data + torch.randn_like(np) * noise)
                break

    def forward(self, x: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (output, feedback)
        """
        if self.num_layers == 0:
            return x, x

        h = x
        if self.l4 is not None:
            h = self.l4(h)
        if self.l23 is not None:
            h = self.l23(h, mask)
        if self.l5 is not None:
            h = self.l5(h)
        feedback = self.l6(h) if self.l6 is not None else h
        h = self.output_norm(h)
        return h, feedback

    def get_pe_activity(self) -> float:
        if self.l23 is not None:
            return self.l23.get_pe_activity()
        return 0.0

    def update_pe_prev(self):
        if self.l23 is not None:
            self.l23.update_pe_prev()

    @property
    def is_mature(self) -> bool:
        return self.num_layers >= self.max_layers

    @property
    def layer_names(self) -> list:
        return self.growth_order[:self.num_layers]


class Neuroblast(SparkColumn):
    """神经母细胞：刚分裂出来的未成熟柱"""
    pass


# ═══════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    d_model, d_ff, n_heads = 64, 256, 4

    print("=== 渐进生长测试 ===")
    col = SparkColumn(d_model, d_ff, n_heads, num_layers=0)
    print(f"初始: layers={col.num_layers}, names={col.layer_names}")

    while col.grow():
        print(f"grow() → layers={col.num_layers}, names={col.layer_names}")

    print(f"\n成熟? {col.is_mature}")

    print("\n=== 前向传播 ===")
    x = torch.randn(2, 10, d_model)
    out, fb = col(x)
    print(f"x: {x.shape} → out: {out.shape}, feedback: {fb.shape}")

    print(f"\nPE 活跃度: {col.get_pe_activity():.6f}")

    nb = Neuroblast(d_model, d_ff, n_heads, num_layers=1)
    print(f"\nNeuroblast: layers={nb.num_layers}, names={nb.layer_names}")

    total_p = sum(p.numel() for p in col.parameters())
    print(f"\n成熟柱参数: {total_p:,}")
