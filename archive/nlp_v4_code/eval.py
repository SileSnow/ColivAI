"""
评估脚本
========
1. 加载最佳模型，计算 Test PPL
2. 交互式 / 演示式文本生成
"""

import os
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sentencepiece as spm

from config import *
from model import TransformerLM
from train import LMDataset


def load_model(device):
    """加载训练好的最佳模型。"""
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

    model_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}，请先运行 train.py")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[✓] 模型加载: {model_path}")
    print(f"    参数量: {model.count_parameters():,}")
    return model


def load_sp():
    """加载 BPE 分词器。"""
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL_PREFIX + ".model")
    return sp


# ─── PPL 评估 ───────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, dataloader, device):
    """计算数据集上的困惑度。"""
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


# ─── 生成 ───────────────────────────────────────────────────

@torch.no_grad()
def generate(model, sp, device, prompt: str, max_tokens: int = 100, temperature: float = 0.8):
    """给定 prompt，自回归生成。"""
    prompt_ids = [BOS_ID] + sp.encode(prompt, out_type=int)
    gen_ids = model.generate(prompt_ids, max_new_tokens=max_tokens, temperature=temperature)
    return sp.decode(gen_ids)


# ─── 入口 ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V4 Baseline 评估")
    parser.add_argument("--mode", choices=["ppl", "generate", "both"], default="both",
                        help="评估模式: ppl / generate / both")
    parser.add_argument("--prompt", type=str, default="The ",
                        help="生成 prompt（仅 generate 模式）")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载
    model = load_model(device)
    sp = load_sp()

    # PPL
    if args.mode in ("ppl", "both"):
        print("\n" + "=" * 50)
        print("  Perplexity 评估")
        print("=" * 50)

        for split in ["train", "valid", "test"]:
            path = os.path.join(DATA_DIR, f"{split}.pt")
            if not os.path.exists(path):
                print(f"  {split}: 跳过（文件不存在）")
                continue
            chunks = torch.load(path)
            dataset = LMDataset(chunks)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            loss, ppl = compute_ppl(model, loader, device)
            print(f"  {split:>6}: Loss={loss:.4f}  PPL={ppl:.2f}")

    # 生成
    if args.mode in ("generate", "both"):
        print("\n" + "=" * 50)
        print("  文本生成")
        print("=" * 50)

        prompts = [
            "The ",
            "It was a dark and stormy night ",
            "The history of the Roman Empire ",
            "Machine learning is ",
        ]

        for prompt in prompts:
            print(f"\n  [Prompt] {prompt}")
            result = generate(model, sp, device, prompt, args.max_tokens, args.temperature)
            print(f"  [生成]   {result}")


if __name__ == "__main__":
    main()
