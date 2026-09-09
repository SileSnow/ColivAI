"""
ALiBi Transformer 语言模型
============================
与 V4 Baseline 唯一区别：
  - 删除 sin/cos 位置编码
  - 手写 MultiheadAttention + ALiBi 线性偏置（硬编码，不可学习）

参数量: ~5.25M（同 Baseline）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── ALiBi Multi-Head Attention ─────────────────────────────

class MultiHeadALiBi(nn.Module):
    """
    手写多头注意力，注入 ALiBi 偏置。

    ALiBi 规则:
      bias[i,j] = -m × |i-j|
      每个头 m 不同，几何级数分配，写死。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # Q, K, V 投影（合并为一个大矩阵）
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ─── ALiBi 斜率（硬编码，不可学习）───
        # m_h = 2^(-2 * (h+1) / nhead) → [0.707, 0.5, 0.354, 0.25] for nhead=4
        # 或者直接用几何级数: m_h = 0.5 / (2^(h))
        # 这里用经典 ALiBi: slopes = [0.5, 0.25, 0.125, 0.0625, ...]
        base = 2.0
        slopes = torch.tensor(
            [1.0 / (base ** (i + 1)) for i in range(nhead)],
            dtype=torch.float,
        )  # shape: (nhead,)
        # 注册为 buffer（不参与训练，但随模型移动 device）
        self.register_buffer("slopes", slopes)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # 1. QKV 投影
        qkv = self.qkv(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, L, D)

        # 2. 拆成多头
        q = q.view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
        k = k.view(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # 3. 注意力分数
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, nhead, L, L)

        # 4. ALiBi 偏置矩阵
        # distance[i,j] = |i-j|
        positions = torch.arange(L, device=x.device, dtype=torch.float)
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))  # (L, L)
        # alibi_bias: (nhead, L, L)
        alibi_bias = -self.slopes.view(self.nhead, 1, 1) * distances.unsqueeze(0)
        # 加到 scores: (B, nhead, L, L) + (1, nhead, L, L)
        scores = scores + alibi_bias.unsqueeze(0)

        # 5. Causal mask
        if causal:
            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=x.device),
                diagonal=1,
            )
            scores = scores + causal_mask  # broadcast over B, nhead

        # 6. Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. 加权求和
        out = torch.matmul(attn_weights, v)  # (B, nhead, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        # 8. 输出投影
        out = self.out_proj(out)
        return out


# ─── Decoder Layer（同 Baseline，但用 MultiHeadALiBi）───────

class DecoderLayerALiBi(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadALiBi(d_model, nhead, dropout)
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


# ─── 完整语言模型（无位置编码）───────────────────────────────

class TransformerLM_ALiBi(nn.Module):
    """
    Decoder-only Transformer with ALiBi.
    无 sin/cos PE —— 位置信息全部由 ALiBi 偏置提供。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        nlayer: int = 6,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # ★ 没有 PositionalEncoding ★
        self.layers = nn.ModuleList(
            [DecoderLayerALiBi(d_model, nhead, d_ff, dropout) for _ in range(nlayer)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # 共享权重
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
        """input_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)"""
        # Token embedding（无 PE）
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        # 直接进 Decoder layers
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self, prompt_ids: list, max_new_tokens: int = 50, temperature: float = 0.8
    ) -> list:
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
            if next_token == 2:  # EOS
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_detail(self) -> dict:
        emb = sum(p.numel() for p in self.token_embedding.parameters())
        layers_total = 0
        for layer in self.layers:
            layers_total += sum(p.numel() for p in layer.parameters())
        head = sum(p.numel() for p in self.lm_head.parameters())
        return {
            "embedding": emb,
            f"{len(self.layers)}_layers": layers_total,
            "lm_head": head,
            "total": emb + layers_total + head,
        }
