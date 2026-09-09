"""
baseline.py — 稳定基线：标准 Transformer + sin/cos + SFT 做 1 步加法
========================================================================
设计原则：
  - 零花活：标准因果 Decoder，标准 sin/cos 绝对位置编码
  - 任务最小：1 步 ± 运算（a+b= 或 a-b=）
  - 训练最稳：纯 SFT teacher forcing，不加 GRPO/RL
  - 可验证：每个 epoch 打印准确率，清晰看到是否学会

一旦这个基线成功 → 再逐步扩展
  Step 1: 1步加法 ✅（本文件）
  Step 2: 连续链 2~8 步
  Step 3: 引入新位置编码 / 新架构想法
  Step 4: 加 GRPO / RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import time
import os
from pathlib import Path

# ============================================================
#  0. 全局配置
# ============================================================

class Config:
    # 模型
    D_MODEL   = 128
    N_LAYERS  = 4
    N_HEADS   = 4
    D_FF      = 512
    MAX_LEN   = 32       # 1步加法很短，32足够
    DROPOUT   = 0.0      # 欠拟合模型不需要 dropout

    # 训练
    BATCH_SIZE   = 64
    LR           = 1e-3
    EPOCHS       = 200
    EARLY_STOP_ACC = 0.99   # 验证准确率 > 99% 早停

    # 数据
    MIN_VAL   = 0
    MAX_VAL   = 99       # 两位数加减
    TRAIN_SAMPLES = 20000
    VAL_SAMPLES   = 2000
    VAL_RATIO = 0.1

    # 保存
    CKPT_DIR = "/workspace/baseline/checkpoints"
    LOG_FILE = "/workspace/baseline/train.log"

    DEVICE = "cpu"

# ============================================================
#  1. 字符级 Tokenizer
# ============================================================

class SimpleTokenizer:
    """最简字符级 tokenizer：0-9, +, -, =, 特殊token"""
    def __init__(self):
        self.specials = ['[PAD]', '[BOS]', '[EOS]']
        self.chars    = list('0123456789+-=')
        self.all_tokens = self.specials + self.chars

        self.stoi = {ch: i for i, ch in enumerate(self.all_tokens)}
        self.itos = {i: ch for i, ch in enumerate(self.all_tokens)}

        self.PAD_IDX = self.stoi['[PAD]']
        self.BOS_IDX = self.stoi['[BOS]']
        self.EOS_IDX = self.stoi['[EOS]']

    @property
    def vocab_size(self):
        return len(self.all_tokens)

    def encode(self, text: str, add_special: bool = True) -> list:
        tokens = [self.stoi[ch] for ch in text if ch in self.stoi]
        if add_special:
            tokens = [self.BOS_IDX] + tokens + [self.EOS_IDX]
        return tokens

    def decode(self, ids: list) -> str:
        return ''.join(self.itos.get(i, '?') for i in ids
                       if i not in (self.PAD_IDX, self.BOS_IDX, self.EOS_IDX))

# ============================================================
#  2. 标准 sin/cos 绝对位置编码
# ============================================================

class SinCosPositionalEncoding(nn.Module):
    """标准 Transformer 位置编码，加到 token embedding 上"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        """x: (B, T, d_model)"""
        return x + self.pe[:x.size(1), :]

# ============================================================
#  3. 标准 Transformer Decoder 层
# ============================================================

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# ============================================================
#  4. 完整 Transformer Decoder 模型
# ============================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinCosPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, return_logits=True):
        """
        input_ids: (B, T) — token ids
        Returns: logits (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # 因果掩码
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=input_ids.device),
            diagonal=1
        )

        x = self.token_embed(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.ln_final(x)
        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, tokenizer, max_new_tokens=10, temperature=0.0):
        """自回归生成，temperature=0 即贪婪"""
        self.eval()
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            if len(generated) > self.max_len:
                break
            inp = torch.tensor([generated[-self.max_len:]], device=next(self.parameters()).device)
            logits = self.forward(inp)
            next_logits = logits[0, -1, :]  # 最后一个位置

            if temperature == 0:
                next_id = next_logits.argmax().item()
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            generated.append(next_id)
            if next_id == tokenizer.EOS_IDX:
                break

        self.train()
        return generated

# ============================================================
#  5. 数据生成 & Dataset
# ============================================================

def make_arithmetic_sample(a, b, op):
    """生成一个算术样本: a+b=result 或 a-b=result"""
    if op == '+':
        result = a + b
    else:
        result = a - b
    text = f"{a}{op}{b}={result}"
    return text, result

class ArithmeticDataset(Dataset):
    def __init__(self, tokenizer, n_samples, min_val, max_val, seed=None):
        self.samples = []
        rng = random.Random(seed)

        for _ in range(n_samples):
            a = rng.randint(min_val, max_val)
            b = rng.randint(min_val, max_val)
            op = rng.choice(['+', '-'])
            text, result = make_arithmetic_sample(a, b, op)
            ids = tokenizer.encode(text, add_special=True)
            self.samples.append((ids, result, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_idx):
    """Pad 到 batch 内最大长度"""
    max_len = max(len(item[0]) for item in batch)
    padded = []
    results = []
    texts = []

    for ids, result, text in batch:
        pad_len = max_len - len(ids)
        padded_ids = ids + [pad_idx] * pad_len
        padded.append(padded_ids)
        results.append(result)
        texts.append(text)

    return torch.tensor(padded, dtype=torch.long), results, texts

# ============================================================
#  6. 训练
# ============================================================

def compute_accuracy(model, dataset, tokenizer, config, n_eval=200):
    """评估准确率：贪婪解码，精确匹配答案"""
    model.eval()
    correct = 0
    total = min(n_eval, len(dataset))

    indices = random.sample(range(len(dataset)), total)
    for idx in indices:
        ids, gt_result, text = dataset[idx]
        # 找到 '=' 的位置作为 prompt 截止点
        decoded = tokenizer.decode(ids)
        eq_pos = decoded.find('=')
        prompt_text = decoded[:eq_pos+1]  # 包含 '='
        prompt_ids = [tokenizer.BOS_IDX] + tokenizer.encode(prompt_text, add_special=False)

        generated = model.generate(prompt_ids, tokenizer, max_new_tokens=10, temperature=0)
        pred_text = tokenizer.decode(generated)
        # 提取 '=' 后面的数字
        eq_idx = pred_text.find('=')
        if eq_idx != -1:
            pred_str = pred_text[eq_idx+1:].strip()
            # 提取数字部分
            import re
            match = re.search(r'-?\d+', pred_str)
            if match:
                try:
                    pred_val = int(match.group())
                    if pred_val == gt_result:
                        correct += 1
                except:
                    pass
    model.train()
    return correct / total if total > 0 else 0.0


def train(config=None):
    if config is None:
        config = Config()

    os.makedirs(config.CKPT_DIR, exist_ok=True)

    # === Init ===
    tokenizer = SimpleTokenizer()
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"特殊token: PAD={tokenizer.PAD_IDX}, BOS={tokenizer.BOS_IDX}, EOS={tokenizer.EOS_IDX}")

    # === Data ===
    train_ds = ArithmeticDataset(tokenizer, config.TRAIN_SAMPLES,
                                  config.MIN_VAL, config.MAX_VAL, seed=42)
    val_ds   = ArithmeticDataset(tokenizer, config.VAL_SAMPLES,
                                  config.MIN_VAL, config.MAX_VAL, seed=123)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer.PAD_IDX))
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer.PAD_IDX))

    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")
    print(f"示例: {train_ds[0][2]} → gt={train_ds[0][1]}")

    # === Model ===
    model = StandardTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        max_len=config.MAX_LEN,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # === Optimizer & Loss ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)

    # === Training loop ===
    print(f"\n{'='*60}")
    print(f"开始训练 — {config.EPOCHS} epochs, LR={config.LR}")
    print(f"{'='*60}\n")

    best_acc = 0.0
    log_lines = []

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_ids, _, _ in train_loader:
            batch_ids = batch_ids.to(config.DEVICE)
            B, T = batch_ids.shape

            # input:  [BOS, a, op, b, =, result, EOS, PAD, ...]
            # target: [a, op, b, =, result, EOS, PAD, ...]  (shifted right by 1)
            logits = model(batch_ids)  # (B, T, vocab_size)
            logits = logits[:, :-1, :].contiguous()  # 去掉最后一个预测
            targets = batch_ids[:, 1:].contiguous()   # 去掉第一个 token

            # 🔑 只对 "=" 及之后的答案 token 算 loss，屏蔽抄题部分
            eq_id = tokenizer.stoi['=']
            masked_targets = targets.clone()
            for b in range(B):
                eq_positions = (batch_ids[b] == eq_id).nonzero(as_tuple=True)[0]
                if len(eq_positions) > 0:
                    eq_pos = eq_positions[0].item()
                    target_start = max(0, eq_pos - 1)  # target 中 "=" 的位置
                    if target_start < masked_targets.shape[1]:
                        masked_targets[b, :target_start] = tokenizer.PAD_IDX

            loss = loss_fn(logits.view(-1, tokenizer.vocab_size), masked_targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0

        # 评估
        train_acc = compute_accuracy(model, train_ds, tokenizer, config, n_eval=100)
        val_acc   = compute_accuracy(model, val_ds, tokenizer, config, n_eval=100)

        # 日志
        lr_now = scheduler.get_last_lr()[0]
        line = (f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"TrainAcc: {train_acc:.1%} | ValAcc: {val_acc:.1%} | "
                f"LR: {lr_now:.2e} | {elapsed:.1f}s")
        print(line, flush=True)
        log_lines.append(line)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config.CKPT_DIR}/best.pth")
            print(f"  ⭐ 新最佳! ValAcc={val_acc:.1%}", flush=True)

        # 早停
        if val_acc >= config.EARLY_STOP_ACC:
            print(f"\n🎉 早停! 验证准确率 {val_acc:.1%} >= {config.EARLY_STOP_ACC:.0%}")
            break

    # === 最终评估 ===
    print(f"\n{'='*60}")
    print("最终评估")
    print(f"{'='*60}")

    # 加载最佳模型
    best_path = f"{config.CKPT_DIR}/best.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=config.DEVICE, weights_only=True))

    model.eval()
    final_acc = compute_accuracy(model, val_ds, tokenizer, config, n_eval=500)
    print(f"最终准确率 (500样本): {final_acc:.1%}")

    # 打印几个生成样例
    print(f"\n生成样例:")
    tokenizer_stoi = tokenizer.stoi  # 用于局部访问
    for _ in range(5):
        idx = random.randint(0, len(val_ds) - 1)
        ids, gt, text = val_ds[idx]
        decoded = tokenizer.decode(ids)
        eq_pos = decoded.find('=')
        prompt_text = decoded[:eq_pos+1]
        prompt_ids = [tokenizer.BOS_IDX] + tokenizer.encode(prompt_text, add_special=False)
        generated = model.generate(prompt_ids, tokenizer, max_new_tokens=10, temperature=0)
        pred_text = tokenizer.decode(generated)
        ok = "✅" if str(gt) in pred_text else "❌"
        print(f"  {ok} 题目: {decoded[:eq_pos]}  →  预测: {pred_text[eq_pos+1:] if eq_pos < len(pred_text) else '?'}  (gt={gt})")

    # 保存日志
    with open(config.LOG_FILE, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\n📝 日志已保存到 {config.LOG_FILE}")

    # 保存最终模型
    torch.save(model.state_dict(), f"{config.CKPT_DIR}/final.pth")
    print(f"💾 模型已保存到 {config.CKPT_DIR}/")

    return model, tokenizer, final_acc


# ============================================================
#  7. 入口
# ============================================================

if __name__ == "__main__":
    config = Config()
    print("=" * 60)
    print("baseline.py — 标准 Transformer + sin/cos + SFT")
    print("=" * 60)
    print(f"d_model={config.D_MODEL}, layers={config.N_LAYERS}, "
          f"heads={config.N_HEADS}, d_ff={config.D_FF}")
    print(f"参数量: ~{sum(p.numel() for p in StandardTransformer(16, 128, 4, 4, 512, 32).parameters()):,}")
    print(f"设备: {config.DEVICE}")
    print()

    model, tokenizer, acc = train(config)
    print(f"\n✅ 训练完成！最终准确率: {acc:.1%}")
