"""
LLaMA 风格 RoPE Transformer 语言模型
=====================================
- RoPE 旋转位置编码
- Pre-RMSNorm（前置归一化）
- SwiGLU FFN
- 无 bias Linear

参数量: vocab=5000 → ~6.02M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── RMSNorm ────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA 同款)"""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * self.weight / rms


# ─── RoPE Attention ─────────────────────────────────────────

class RoPEMultiHeadAttention(nn.Module):
    """
    RoPE 旋转位置编码 + 手写多头注意力。
    RoPE: 对 Q, K 的每对维度按位置旋转，使 Q·K 天然依赖相对位置。
    """

    def __init__(self, d_model: int, nhead: int, max_len: int = 600, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.max_len = max_len

        # Q, K, V 投影（无 bias）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # ─── RoPE: 预计算 cos/sin 表 ───
        # theta_i = 1 / 10000^(2i/d_k)  for i in [0, d_k/2)
        theta = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        # (d_k/2,)

        # 每个位置 m 的旋转角: m * theta  → (max_len, d_k/2)
        positions = torch.arange(max_len).float().unsqueeze(1)  # (max_len, 1)
        angles = positions * theta.unsqueeze(0)                  # (max_len, d_k/2)

        # cos, sin 表: (1, 1, max_len, d_k/2) 方便广播
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将后半维度取负并与前半交换: [x1,x2,x3,x4] → [-x2,x1,-x4,x3]"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        对最后两维应用 RoPE 旋转。
        x: (B, nhead, L, d_k)
        """
        B, H, L, D = x.shape
        cos = self.cos[:, :, offset:offset + L, :]  # (1, 1, L, d_k/2)
        sin = self.sin[:, :, offset:offset + L, :]

        # 每对维度共享同一角度，repeat 到完整 d_k
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # (1, 1, L, d_k)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        return x * cos + self._rotate_half(x) * sin

    def forward(self, x: torch.Tensor, causal: bool = True, offset: int = 0) -> torch.Tensor:
        B, L, D = x.shape

        # Pre-norm 已在外面做好，这里直接投影
        q = self.q_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # ★ RoPE 旋转 Q 和 K
        q = self._apply_rope(q, offset)
        k = self._apply_rope(k, offset)

        # Q·K^T / √d_k
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        if causal:
            mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ─── SwiGLU FFN ─────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU: gate(x) * up(x) → down
    参数量 ≈ 标准 FFN（d_ff_intermediate = 688 时）
    """
    def __init__(self, d_model: int, d_inter: int = 688, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inter, bias=False)
        self.up_proj   = nn.Linear(d_model, d_inter, bias=False)
        self.down_proj = nn.Linear(d_inter, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # SiLU = Swish
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# ─── Decoder Layer (LLaMA 风格) ─────────────────────────────

class DecoderLayerRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_inter: int = 688, max_len: int = 600, dropout: float = 0.0):
        super().__init__()
        # Pre-norm: RMSNorm 在 attention 和 FFN 之前
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm  = RMSNorm(d_model)
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, max_len, dropout)
        self.ffn        = SwiGLUFFN(d_model, d_inter, dropout)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # Pre-norm + residual
        x = x + self.self_attn(self.attn_norm(x), causal=True, offset=offset)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── 完整 LLaMA 风格语言模型 ────────────────────────────────

class TransformerLM_RoPE(nn.Module):
    """
    LLaMA 风格 Decoder-only:
    - Token Embedding → RMSNorm → 6×DecoderLayerRoPE → RMSNorm → LM Head
    - 无位置编码嵌入（RoPE 在 Attention 内部）
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        nhead: int = 4,
        nlayer: int = 6,
        d_inter: int = 688,
        max_len: int = 600,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.nlayer = nlayer
        self.nhead = nhead

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.layers = nn.ModuleList(
            [DecoderLayerRoPE(d_model, nhead, d_inter, max_len, dropout) for _ in range(nlayer)]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, input_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # 无 sin/cos PE，RoPE 在 Attention 内部处理位置
        x = self.token_embedding(input_ids)  # 不用 *√d，LLaMA 风格
        for layer in self.layers:
            x = layer(x, offset=offset)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: list, max_new_tokens: int = 50, temperature: float = 0.8) -> list:
        self.eval()
        device = next(self.parameters()).device
        generated = list(prompt_ids)

        offset = 0
        for i in range(max_new_tokens):
            ctx = generated[-self.max_len:]
            inp = torch.tensor([ctx], device=device)
            # 第一个 batch 用 offset=0, 之后每个 token offset 递增
            logits = self(inp, offset=offset)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            offset += 1
            if next_token == 2:
                break
        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
