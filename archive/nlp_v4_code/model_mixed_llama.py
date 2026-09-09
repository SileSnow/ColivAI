"""
Mixed-LLaMA: LLaMA 架构 + tanh/sech² 混合 PE
=============================================
结构: Pre-RMSNorm + SwiGLU + 无bias (LLaMA 风格)
PE:   sin/cos 绝对 + tanh/sech² 可学习相对偏置 (Mixed 风格)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── RMSNorm ────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * self.weight / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)


# ─── sin/cos 绝对位置编码 ───────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 600, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ─── tanh/sech² 混合 PE Attention ───────────────────────────

class MixedPEAttention(nn.Module):
    """手写多头注意力 + tanh/sech² 可学习相对偏置 (LLaMA 风格: 无bias, 分离 QKV)"""

    def __init__(self, d_model: int, nhead: int, max_len: int = 600, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.max_len = max_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 可学习参数
        self.w_param = nn.Parameter(torch.ones(nhead))
        self.v_param = nn.Parameter(torch.full((nhead,), 0.02))
        self.log_tau = nn.Parameter(torch.zeros(nhead))

        # 预计算距离矩阵
        self.register_buffer(
            "distances",
            torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            - torch.arange(max_len, dtype=torch.float).unsqueeze(0),
        )

    def _compute_bias(self, L: int) -> torch.Tensor:
        d = self.distances[:L, :L]
        d_abs = torch.abs(d)

        tau = torch.exp(self.log_tau).view(self.nhead, 1, 1)
        tau_d = tau * d_abs.unsqueeze(0)
        sech2 = 1.0 / (torch.cosh(torch.clamp(tau_d, max=85.0)) ** 2)

        v = self.v_param.view(self.nhead, 1, 1)
        w = self.w_param.view(self.nhead, 1, 1)
        tanh_term = torch.tanh(v * d_abs.unsqueeze(0))

        return w * tanh_term + sech2

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        bias = self._compute_bias(L)
        scores = scores + bias.unsqueeze(0)

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
    def __init__(self, d_model: int, d_inter: int = 688, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inter, bias=False)
        self.up_proj   = nn.Linear(d_model, d_inter, bias=False)
        self.down_proj = nn.Linear(d_inter, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─── Decoder Layer (LLaMA + tanh/sech²) ─────────────────────

class DecoderLayerMixedLLaMA(nn.Module):
    def __init__(self, d_model, nhead, d_inter=688, max_len=600, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm  = RMSNorm(d_model)
        self.self_attn = MixedPEAttention(d_model, nhead, max_len, dropout)
        self.ffn        = SwiGLUFFN(d_model, d_inter, dropout)

    def forward(self, x):
        x = x + self.self_attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── 完整模型 ────────────────────────────────────────────────

class TransformerLM_MixedLLaMA(nn.Module):
    """LLaMA 架构 + tanh/sech² PE"""

    def __init__(self, vocab_size=5000, d_model=256, nhead=4, nlayer=6,
                 d_inter=688, max_len=600, dropout=0.0, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayerMixedLLaMA(d_model, nhead, d_inter, max_len, dropout)
            for _ in range(nlayer)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.d_model**-0.5)

    def forward(self, input_ids):
        x = self.pos_encoding(self.token_embedding(input_ids) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=0.8):
        self.eval()
        device = next(self.parameters()).device
        gen = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = gen[-self.max_len:]
            logits = self(torch.tensor([ctx], device=device))
            probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).item()
            gen.append(tok)
            if tok == 2: break
        return gen

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
