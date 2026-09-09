"""
Decoder-only Transformer 语言模型
==================================
- Token Embedding（与 LM Head 共享权重）
- sin/cos 冻结位置编码
- 6 层 Decoder（self-attn + FFN）
- 参数量: ~5.24M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 位置编码（sin/cos，冻结）────────────────────────────────

class PositionalEncoding(nn.Module):
    """标准 sin/cos 绝对位置编码，非可学习（register_buffer）。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# ─── Decoder Layer ───────────────────────────────────────────

class DecoderLayer(nn.Module):
    """一层 Transformer Decoder：Masked Self-Attention + FFN。"""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ─── 完整语言模型 ────────────────────────────────────────────

class TransformerLM(nn.Module):
    """Decoder-only Transformer 语言模型。

    参数量 (VOCAB=2000, d_model=256, nhead=4, nlayer=6, d_ff=1024):
      Token Embedding: 2000 x 256 =   512,000
      每层:
        Self-Attn (QKV+O): 4 x 256^2     =   262,144
        FFN (W1+W2):       2 x 256x1024  =   524,288
        LayerNorm x2:       2 x 2x256    =     1,024
      6 层: 6 x 787,456 = 4,724,736
      LM Head: 共享 embedding (0)
      ─────────────────────────────────────
      总计: ~5,236,736
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
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(nlayer)]
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

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)"""
        batch, seq_len = input_ids.shape

        mask = self._causal_mask(seq_len, input_ids.device)

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.out_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self, prompt_ids: list, max_new_tokens: int = 50, temperature: float = 0.8
    ) -> list:
        """自回归生成 token 序列。"""
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
        for i, layer in enumerate(self.layers):
            n = sum(p.numel() for p in layer.parameters())
            layers_total += n
        head = sum(p.numel() for p in self.lm_head.parameters())
        return {
            "embedding": emb,
            "6_layers": layers_total,
            "lm_head": head,
            "total": emb + layers_total + head,
        }
