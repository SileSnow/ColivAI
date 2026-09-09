"""
训练脚本
========
加载数据 → 构建模型 → 训练 → 保存最佳模型。
"""

import os
import sys
import math
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import *
from model import TransformerLM


# ─── 数据集 ─────────────────────────────────────────────────

class LMDataset(torch.utils.data.Dataset):
    """将切好的块变成 (input, target)，target 右移一位。"""

    def __init__(self, chunks: torch.Tensor):
        """
        chunks: (num_chunks, SEQ_LEN)
        每个块: input = chunk[:-1], target = chunk[1:]
        """
        self.inputs = chunks[:, :-1]   # (N, SEQ_LEN-1)
        self.targets = chunks[:, 1:]   # (N, SEQ_LEN-1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ─── 学习率调度（线性热身 + CosineAnnealing）─────────────────

def get_lr(epoch: int, step: int, steps_per_epoch: int):
    """返回当前步骤的学习率。"""
    global_step = epoch * steps_per_epoch + step
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = EPOCHS * steps_per_epoch

    if global_step < warmup_steps:
        # 线性热身
        return LR * global_step / warmup_steps
    else:
        # Cosine 衰减
        progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
        return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))


# ─── 训练一个 epoch ─────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 调整 LR
        lr = get_lr(epoch, batch_idx, len(dataloader))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()

        logits = model(inputs)  # (B, seq-1, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=PAD_ID,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

        if batch_idx % 50 == 0:
            ppl = math.exp(loss.item())
            print(f"  [Epoch {epoch:2d} | Batch {batch_idx:4d}] "
                  f"Loss={loss.item():.4f}  PPL={ppl:.2f}  LR={lr:.2e}")

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ─── 评估 ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=PAD_ID,
        )

        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ─── 生成演示 ────────────────────────────────────────────────

@torch.no_grad()
def demo_generation(model, sp, device, prompt: str = "The ", max_tokens: int = 50):
    """用训练中的模型做一次生成演示。"""
    model.eval()
    prompt_ids = [BOS_ID] + sp.encode(prompt, out_type=int)
    gen_ids = model.generate(prompt_ids, max_new_tokens=max_tokens, temperature=GEN_TEMPERATURE)
    gen_text = sp.decode(gen_ids)
    return gen_text


# ─── 主训练逻辑 ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  V4 Baseline — 训练")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[🖥️ ] 设备: {device}")

    # 加载数据
    print("[📂] 加载数据...")
    train_chunks = torch.load(os.path.join(DATA_DIR, "train.pt"))
    valid_chunks = torch.load(os.path.join(DATA_DIR, "valid.pt"))
    print(f"    训练: {train_chunks.shape} 块")
    print(f"    验证: {valid_chunks.shape} 块")

    train_dataset = LMDataset(train_chunks)
    valid_dataset = LMDataset(valid_chunks)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    每 epoch: {len(train_loader)} 步")

    # 构建模型
    print("[🏗️ ] 构建模型...")
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        nlayer=NLAYER,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        pad_id=PAD_ID,
    ).to(device)

    total_params = model.count_parameters()
    print(f"    参数量: {total_params:,}")
    detail = model.count_parameters_detail()
    for k, v in detail.items():
        print(f"      {k}: {v:,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    # 加载 BPE（用于生成演示）
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL_PREFIX + ".model")

    # 训练循环
    best_val_ppl = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # 训练
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, epoch, device)

        # 验证
        val_loss, val_ppl = evaluate(model, valid_loader, device)

        t1 = time.time()
        elapsed = t1 - t0

        print(f"\n{'='*50}")
        print(f"  Epoch {epoch:2d} 完成 | "
              f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f} | "
              f"耗时: {elapsed:.0f}s")
        print(f"{'='*50}\n")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
        })

        # 保存最佳模型
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            print(f"  [💾] 保存最佳模型 (Val PPL: {val_ppl:.2f})\n")

        # 生成演示
        if epoch % GEN_EVERY_N_EPOCHS == 0:
            demo = demo_generation(model, sp, device)
            print(f"  [🎤] 生成演示:\n    {demo}\n")

    # 保存训练历史
    with open(os.path.join(LOG_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[✅] 训练完成！最佳 Val PPL: {best_val_ppl:.2f}")
    print(f"    模型保存至: {os.path.join(MODEL_DIR, 'best_model.pt')}")
    print(f"    训练历史:   {os.path.join(LOG_DIR, 'history.json')}")


if __name__ == "__main__":
    main()
