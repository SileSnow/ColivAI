"""
混合位置编码 Transformer 语言模型
==================================
sin/cos 绝对 PE + 可学习 tanh/sech² 相对偏置
每头 3 个参数 (w, v, log_tau) × 4头 × 6层 = 72 个可学习 PE 参数

与 V4 Baseline/ALiBi 区别:
  - Baseline: sin/cos 绝对 PE（无相对偏置）
  - ALiBi: 固定线性偏置（无学习）
  - Mixed: sin/cos + 可学习 tanh/sech²（本文件）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── sin/cos 位置编码（同 Baseline）─────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 600, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ─── 混合 PE Attention（sin/cos + 可学习 tanh/sech²）────────

class MixedPEAttention(nn.Module):
    """
    手写多头注意力，注入 tanh/sech² 可学习相对偏置。

    偏置公式:
      rel_bias(d) = w · tanh(v · d) + sech²(exp(log_tau) · d)

    参数 (每头):
      w:       偏置幅度   → 初始 ~1.0
      v:       tanh 斜率  → 初始 ~0.02
      log_tau: sech² 带宽 → 初始 0.0 (τ=1.0)
    """

    def __init__(self, d_model: int, nhead: int, max_len: int = 600, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.max_len = max_len

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ─── 可学习相对偏置参数 ───
        # 每个头 3 个参数: w, v, log_tau
        # 初始化: w~1.0 (大 → 偏置显著), v~0.02 (小 → tanh 缓慢激活)
        self.w_param = nn.Parameter(torch.ones(nhead))           # ~1.0
        self.v_param = nn.Parameter(torch.full((nhead,), 0.02))  # ~0.02
        self.log_tau = nn.Parameter(torch.zeros(nhead))          # τ = exp(0) = 1.0

        # 预计算距离矩阵（只算一次）
        self.register_buffer(
            "distances",
            torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            - torch.arange(max_len, dtype=torch.float).unsqueeze(0),
            # (max_len, max_len), 负值表示 j > i, 会被 causal mask 盖掉
        )

    def _compute_bias(self, L: int) -> torch.Tensor:
        """
        计算 ALiBi 风格的可学习偏置矩阵。
        返回: (nhead, L, L)
        """
        # 截取需要的距离矩阵
        d = self.distances[:L, :L]  # (L, L)
        d_abs = torch.abs(d)        # |i-j|

        # sech² 部分: 需要 clamp 防止 cosh 溢出
        tau = torch.exp(self.log_tau).view(self.nhead, 1, 1)  # (nhead, 1, 1)
        # cosh(τ·d) 在 τ·d > 85 时溢出 float32
        tau_d = tau * d_abs.unsqueeze(0)  # (nhead, L, L)
        tau_d_clamped = torch.clamp(tau_d, max=85.0)
        sech2 = 1.0 / (torch.cosh(tau_d_clamped) ** 2)  # (nhead, L, L)

        # tanh 部分
        v = self.v_param.view(self.nhead, 1, 1)          # (nhead, 1, 1)
        tanh_term = torch.tanh(v * d_abs.unsqueeze(0))    # (nhead, L, L)
        w = self.w_param.view(self.nhead, 1, 1)           # (nhead, 1, 1)

        bias = w * tanh_term + sech2  # (nhead, L, L)

        # d=0 时（自己看自己）偏置设 0（不加额外偏置）
        # 实际上 tanh(0)=0, sech²(0)=1.0，所以 self-bias=1.0
        # 这相当于给「当前位置」额外的注意力权重。
        # 保留这个设计——和原 verify3.2/3.3 一致。

        return bias

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # 1. QKV 投影
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. 多头
        q = q.view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
        k = k.view(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # 3. 分数
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, nhead, L, L)

        # 4. 可学习相对偏置
        bias = self._compute_bias(L)  # (nhead, L, L)
        scores = scores + bias.unsqueeze(0)

        # 5. Causal mask
        if causal:
            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=x.device), diagonal=1
            )
            scores = scores + causal_mask

        # 6. Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. 加权求和
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # 8. 输出投影
        return self.out_proj(out)

    def get_tanh_params(self) -> dict:
        """调试用：返回 tanh/sech² 参数的当前值。"""
        return {
            "w": self.w_param.detach().tolist(),
            "v": self.v_param.detach().tolist(),
            "tau": torch.exp(self.log_tau).detach().tolist(),
        }


# ─── Decoder Layer ──────────────────────────────────────────

class DecoderLayerMixed(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, max_len: int = 600, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MixedPEAttention(d_model, nhead, max_len, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, causal=True)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ─── 完整语言模型 ────────────────────────────────────────────

class TransformerLM_Mixed(nn.Module):
    """
    sin/cos 绝对 PE + 可学习 tanh/sech² 相对偏置。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        nlayer: int = 6,
        d_ff: int = 1024,
        max_len: int = 600,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayerMixed(d_model, nhead, d_ff, max_len, dropout) for _ in range(nlayer)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: list, max_new_tokens: int = 50, temperature: float = 0.8) -> list:
        self.eval()
        device = next(self.parameters()).device
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = generated[-self.max_len :]
            inp = torch.tensor([ctx], device=device)
            logits = self(inp)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if next_token == 2:
                break
        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_detail(self) -> dict:
        emb = sum(p.numel() for p in self.token_embedding.parameters())
        layers_total = 0
        pe_params = 0
        for layer in self.layers:
            layers_total += sum(p.numel() for p in layer.parameters())
            pe_params += sum(p.numel() for p in layer.self_attn.w_param)
            pe_params += sum(p.numel() for p in layer.self_attn.v_param)
            pe_params += sum(p.numel() for p in layer.self_attn.log_tau)
        head = sum(p.numel() for p in self.lm_head.parameters())
        return {
            "embedding": emb,
            f"{len(self.layers)}_layers": layers_total,
            "PE_params": pe_params,
            "lm_head": head,
            "total": emb + layers_total + head,
        }
