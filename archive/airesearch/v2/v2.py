"""
Transformer + CoT 学会一百以内加减法，并泛化到更高位数
=====================================================
训练：0~100 加减法，带思维链 (Chain-of-Thought)
测试：泛化到 3~6 位数加减法

数学语言（无自然语言）：
  加法：da+db+ci→do↑co   (carry in → digit out ↑ carry out)
  减法：da-db-bi→do↑bo   (borrow in → digit out ↑ borrow out)
  最终：=answer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import math
import random
import time
import os

# ============================================================
#  1. Tokenizer
# ============================================================

# 字符集：数字 + 运算符号 + CoT 符号
CHARS = "0123456789+-=\n"
# 索引分配：0=PAD, 1=BOS, 2=EOS, 3..22=字符
PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2


class CharTokenizer:
    """字符级 tokenizer，用于数学 CoT 语言"""
    def __init__(self):
        self.char2idx = {c: i + 3 for i, c in enumerate(CHARS)}
        self.char2idx['[PAD]'] = PAD_IDX
        self.char2idx['[BOS]'] = BOS_IDX
        self.char2idx['[EOS]'] = EOS_IDX
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        tokens = []
        if add_special:
            tokens.append(BOS_IDX)
        for c in text:
            tokens.append(self.char2idx.get(c, self.char2idx['\n']))
        if add_special:
            tokens.append(EOS_IDX)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return ''.join(self.idx2char.get(t, '?') for t in tokens)


# ============================================================
#  2. CoT 数据生成
# ============================================================

def generate_add_cot(a: int, b: int) -> str:
    """
    生成 a + b 的思维链。
    格式（从右向左逐列）：
        a+b=
        da+db+ci→do↑co
        ...
        =result
    其中 ci=carry_in, do=digit_out, co=carry_out
    """
    a_str, b_str = str(a), str(b)
    n = max(len(a_str), len(b_str)) + 1         # 多一列处理可能的进位
    a_pad, b_pad = a_str.zfill(n), b_str.zfill(n)

    lines = [f"{a}+{b}="]
    carry_in = 0

    for i in range(n - 1, -1, -1):
        da, db = int(a_pad[i]), int(b_pad[i])
        total = da + db + carry_in
        digit_out = total % 10
        carry_out = total // 10
        lines.append(f"{da}+{db}+{carry_in}→{digit_out}↑{carry_out}")
        carry_in = carry_out

    lines.append(f"={a + b}")
    return "\n".join(lines)


def generate_sub_cot(a: int, b: int) -> str:
    """
    生成 a - b 的思维链。
    格式（从右向左逐列）：
        a-b=
        da-db-bi→do↑bo
        ...
        =result
    其中 bi=borrow_in, do=digit_out, bo=borrow_out
    当 a<b 时：a-b = -(b-a)
    """
    # --- 处理 a < b（负结果） ---
    if a < b:
        inner = generate_sub_cot(b, a)          # 递归：先生成 b-a 的 CoT
        inner_lines = inner.split("\n")
        # inner_lines[0] = "b-a="
        # inner_lines[1:-1] = CoT 步骤
        # inner_lines[-1] = "=(b-a)"
        # 我们需要输出 "a-b=\n-(b-a)\n[步骤]\n=-(b-a)"
        result_val = -(b - a)
        cot_lines = [f"{a}-{b}=", f"-({b}-{a})"]
        cot_lines.extend(inner_lines[1:-1])      # 中间的 CoT 步骤
        cot_lines.append(f"={result_val}")
        return "\n".join(cot_lines)

    # --- 正常 a >= b ---
    a_str, b_str = str(a), str(b)
    n = max(len(a_str), len(b_str))              # 减法结果不会比被减数更多位
    a_pad, b_pad = a_str.zfill(n), b_str.zfill(n)

    lines = [f"{a}-{b}="]
    borrow_in = 0

    for i in range(n - 1, -1, -1):
        da, db = int(a_pad[i]), int(b_pad[i])
        diff = da - db - borrow_in
        if diff >= 0:
            digit_out = diff
            borrow_out = 0
            lines.append(f"{da}-{db}-{borrow_in}→{digit_out}↑{borrow_out}")
            borrow_in = 0
        else:
            digit_out = diff + 10
            borrow_out = 1
            lines.append(f"{da}-{db}-{borrow_in}→{digit_out}↑{borrow_out}")
            borrow_in = 1

    lines.append(f"={a - b}")
    return "\n".join(lines)


# ============================================================
#  3. 数据集
# ============================================================

class ArithmeticCoTDataset(Dataset):
    """
    生成加减法 CoT 数据。
    op='add' 只生成加法，'sub' 只生成减法，'both' 生成混合。
    """
    def __init__(self, max_val: int = 100, num_samples: int = 10000,
                 op: str = 'both', seed: int = 42):
        self.tokenizer = CharTokenizer()
        rng = np.random.RandomState(seed)

        samples = []
        # ----- 加法 -----
        if op in ('add', 'both'):
            # 全部组合可能太多，随机采样
            n_add = num_samples // 2 if op == 'both' else num_samples
            for _ in range(n_add):
                a = rng.randint(0, max_val + 1)
                b = rng.randint(0, max_val + 1)
                samples.append(generate_add_cot(a, b))

        # ----- 减法 -----
        if op in ('sub', 'both'):
            n_sub = num_samples // 2 if op == 'both' else num_samples
            for _ in range(n_sub):
                a = rng.randint(0, max_val + 1)
                b = rng.randint(0, max_val + 1)
                samples.append(generate_sub_cot(a, b))

        rng.shuffle(samples)

        # 编码所有样本
        self.data = []
        self.mask = []          # 1 = 参与 loss 计算（CoT+答案）, 0 = 不参与（prompt）
        self.max_len = 0

        for sample in samples:
            tokens = self.tokenizer.encode(sample)
            # prompt 部分："a+b=\n" 或 "a-b=\n" 或 "a-b=\n-(...)\n"
            # 找到第一个 "=" 之后的内容作为 CoT 起点
            # 更准确：prompt 结束于第一个 "\n"（它是输入和 CoT 的分隔）
            text_before_cot = sample.split("\n")[0] + "\n"   # 如 "47+58=\n"
            prompt_len = len(self.tokenizer.encode(text_before_cot, add_special=True)) - 1  # 去掉 EOS

            # target 序列：从 prompt 之后开始
            target_tokens = tokens[prompt_len:]
            # 补 PAD 到统一长度（稍后统一处理）
            self.data.append(tokens)
            # mask: prompt 部分为 0，CoT+答案部分为 1
            m = [0] * prompt_len + [1] * (len(tokens) - prompt_len)
            self.mask.append(m)

        # 统一 padding 到最长序列
        self.max_len = max(len(d) for d in self.data)
        for i in range(len(self.data)):
            pad_len = self.max_len - len(self.data[i])
            self.data[i] = self.data[i] + [PAD_IDX] * pad_len
            self.mask[i] = self.mask[i] + [0] * pad_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        mask = torch.tensor(self.mask[idx], dtype=torch.bool)
        return tokens, mask


# ============================================================
#  4. Transformer 模型（Decoder-only）
# ============================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerCoT(nn.Module):
    """
    轻量级 Transformer Decoder，用于 CoT 算术推理。
    参数量 ≈ vocab×d_model + n_layers×(4×d_model² + 8×d_model×n_heads + 2×d_model×d_ff)
    """
    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, d_ff: int = 512,
                 max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_mask=None):
        """
        x: (batch, seq_len)  token ids
        src_mask: (batch, seq_len)  bool mask, True=参加注意力
        返回: (batch, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        # 因果掩码（三角形）
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )

        # Embedding + Position
        h = self.token_embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)
        h = self.dropout(h)

        # 逐层 Transformer Decoder
        for layer in self.layers:
            h = layer(h, h, tgt_mask=causal_mask, tgt_key_padding_mask=~src_mask if src_mask is not None else None)

        h = self.norm(h)
        logits = self.output_proj(h)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, prompt: str, tokenizer: CharTokenizer,
                 max_gen_len: int = 200, temperature: float = 0.0) -> str:
        """
        给定 prompt（如 "47+58="），自回归生成完整 CoT + 答案。
        temperature=0 贪婪解码，>0 采样。
        """
        self.eval()
        device = next(self.parameters()).device

        # 编码 prompt
        input_ids = torch.tensor(tokenizer.encode(prompt, add_special=True), dtype=torch.long).unsqueeze(0).to(device)

        generated = []
        for _ in range(max_gen_len):
            # 创建 mask（所有 token 都参与注意力）
            src_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)

            # Forward
            logits = self(input_ids, src_mask)

            # 取最后一个位置的 logits
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # 温度采样 / 贪婪
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = next_logits.argmax().item()

            # 检查结束
            if next_token == EOS_IDX:
                break

            generated.append(next_token)

            # 拼接
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)

            # 检查是否生成了答案行 "=数字"
            partial = tokenizer.decode(generated)
            # 如果包含 "\n="，说明答案已出，可以继续到 EOS
            # 简单策略：如果最后生成的是 '\n'，检查是否已完成
            if next_token == tokenizer.char2idx.get('\n', -1):
                if partial.rstrip("\n").endswith("="):
                    pass  # 答案行可能还没完

        return tokenizer.decode(generated)


# ============================================================
#  5. 训练
# ============================================================

def train_model(model, dataloader, epochs=30, lr=1e-3, device='cpu'):
    """训练 Transformer CoT 模型"""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        for batch_idx, (tokens, mask) in enumerate(dataloader):
            tokens, mask = tokens.to(device), mask.to(device)

            # 输入 = tokens[:, :-1], 目标 = tokens[:, 1:]
            # mask[:, :-1] 对应输入，mask[:, 1:] 对应目标
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            tgt_mask = mask[:, 1:]            # 哪些目标 token 参与 loss

            # 前向
            logits = model(inp, mask[:, :-1])  # (batch, seq_len-1, vocab_size)

            # Loss (只计算 tgt_mask 为 True 的位置)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction='none'
            )  # (batch * (seq_len-1),)
            loss = loss.reshape_as(tgt)        # (batch, seq_len-1)
            loss = loss * tgt_mask             # 只保留 CoT 部分的 loss
            num_tokens = tgt_mask.sum().item()
            if num_tokens == 0:
                continue
            batch_loss = loss.sum() / num_tokens

            # 反向
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item() * num_tokens
            total_tokens += num_tokens

        scheduler.step()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        elapsed = time.time() - start_time
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  |  Loss: {avg_loss:.6f}  |  "
                  f"Time: {elapsed:.1f}s  |  LR: {scheduler.get_last_lr()[0]:.2e}")

    return model


# ============================================================
#  6. 评估与泛化测试
# ============================================================

@torch.no_grad()
def evaluate_model(model, tokenizer, device='cpu', num_tests=10, max_val=100):
    """在训练范围内（0~max_val）随机测试"""
    model.eval()
    print(f"\n========== 范围内测试 (0~{max_val}) ==========")
    correct = 0
    for i in range(num_tests):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        op = random.choice(['+', '-'])
        if op == '+':
            prompt = f"{a}+{b}="
            true_answer = a + b
        else:
            prompt = f"{a}-{b}="
            true_answer = a - b

        output = model.generate(prompt, tokenizer)
        # 提取答案：找最后出现的 "=数字"
        import re
        match = re.search(r'=(-?\d+)', output)
        pred_answer = int(match.group(1)) if match else None

        ok = pred_answer == true_answer
        if ok:
            correct += 1
        status = "✅" if ok else "❌"
        print(f"  {prompt}{output}")
        print(f"    True: {true_answer}  |  Pred: {pred_answer}  {status}")

    acc = correct / num_tests * 100
    print(f"  准确率: {acc:.1f}%")
    return acc


@torch.no_grad()
def test_generalization(model, tokenizer, device='cpu'):
    """泛化测试：更高位数的加减法"""
    model.eval()
    print(f"\n========== 泛化测试（更高位数）==========")

    test_cases = [
        # 3位数
        ("123+456=", 123+456),
        ("789+211=", 789+211),
        ("999+1=",   999+1),
        ("500-237=", 500-237),
        ("100-1=",   100-1),
        ("321-123=", 321-123),
        # 4位数
        ("1234+5678=", 1234+5678),
        ("9999+1=",    9999+1),
        ("8000-1=",    8000-1),
        ("5432-1234=", 5432-1234),
        # 5位数
        ("12345+54321=", 12345+54321),
        ("99999+1=",     99999+1),
        ("50000-1=",     50000-1),
        # 6位数
        ("123456+654321=",  123456+654321),
        ("999999+1=",       999999+1),
        # 混合位数
        ("7+123=",   7+123),
        ("99+999=",  99+999),
        ("1000-1=",  1000-1),
        ("1000-999=", 1000-999),
    ]

    correct = 0
    for prompt, true_answer in test_cases:
        output = model.generate(prompt, tokenizer, max_gen_len=300)
        import re
        match = re.search(r'=(-?\d+)', output)
        pred_answer = int(match.group(1)) if match else None

        ok = pred_answer == true_answer
        if ok:
            correct += 1
        status = "✅" if ok else "❌"
        print(f"  {prompt}{output}")
        print(f"    True: {true_answer}  |  Pred: {pred_answer}  {status}")

    acc = correct / len(test_cases) * 100
    print(f"\n  泛化准确率: {correct}/{len(test_cases)} = {acc:.1f}%")
    return acc


# ============================================================
#  7. 主流程
# ============================================================

def main():
    # ---------- 配置 ----------
    D_MODEL     = 128        # 嵌入维度
    N_LAYERS    = 4          # Transformer 层数
    N_HEADS     = 4          # 注意力头数
    D_FF        = 512        # FFN 中间维度
    MAX_LEN     = 256        # 最大序列长度
    DROPOUT     = 0.1

    BATCH_SIZE  = 64
    EPOCHS      = 40
    LR          = 5e-4
    NUM_SAMPLES = 12000      # 训练样本数（加减各半）
    MAX_VAL     = 100        # 训练数据范围

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    # ---------- 数据 ----------
    print("\n=== 生成 CoT 数据… ===")
    dataset = ArithmeticCoTDataset(max_val=MAX_VAL, num_samples=NUM_SAMPLES,
                                   op='both', seed=42)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    tokenizer = dataset.tokenizer
    print(f"  样本数: {len(dataset)}")
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  最大序列长度: {dataset.max_len}")
    print(f"  示例:\n{generate_add_cot(47, 58)}\n")
    print(f"  示例:\n{generate_sub_cot(83, 47)}\n")
    print(f"  示例(负结果):\n{generate_sub_cot(23, 47)}\n")

    # ---------- 模型 ----------
    print("=== 构建模型… ===")
    model = TransformerCoT(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # ---------- 训练 ----------
    print("\n=== 训练 ===")
    train_model(model, dataloader, epochs=EPOCHS, lr=LR, device=DEVICE)

    # ---------- 保存 ----------
    torch.save(model.state_dict(), "/workspace/transformer_cot.pth")
    print("\n模型已保存: /workspace/transformer_cot.pth")

    # ---------- 范围内评估 ----------
    evaluate_model(model, tokenizer, device=DEVICE, num_tests=10, max_val=MAX_VAL)

    # ---------- 泛化测试 ----------
    test_generalization(model, tokenizer, device=DEVICE)

    print("\n=== 完成！===")


if __name__ == "__main__":
    main()
