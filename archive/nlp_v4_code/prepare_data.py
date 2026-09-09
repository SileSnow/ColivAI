"""
数据准备：WikiText-2 → 训练 BPE 分词器 → 编码并分块
===================================================
运行一次即可，产物存入 data/ 目录。
需要网络 + sentencepiece + datasets。
"""

import os
import sys

import torch
import sentencepiece as spm

from config import (
    DATA_DIR, VOCAB_SIZE, SEQ_LEN,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
    BPE_MODEL_PREFIX,
)


# ─── 第 0 步：下载 WikiText-2（HuggingFace datasets）─────────

def download_wikitext2():
    """使用 HuggingFace datasets 下载 WikiText-2，保存为纯文本。"""
    from datasets import load_dataset

    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # WikiText-2 的分集名: "train", "validation", "test"
    split_map = {
        "train": ("train", "wiki.train.raw"),
        "valid": ("validation", "wiki.valid.raw"),
        "test": ("test", "wiki.test.raw"),
    }
    files = {k: os.path.join(raw_dir, v[1]) for k, v in split_map.items()}

    # 如果文本文件已存在，跳过
    if all(os.path.exists(f) for f in files.values()):
        print("[✓] WikiText-2 文本已存在，跳过下载。")
        return files

    print("[↓] 通过 HuggingFace datasets 加载 WikiText-2...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    for key, (split_name, filename) in split_map.items():
        path = files[key]
        print(f"    保存 {key} → {path}")
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset[split_name]:
                f.write(item["text"] + "\n")

    print("[✓] WikiText-2 下载完成。")
    return files


# ─── 第 1 步：训练 BPE ──────────────────────────────────────

def train_bpe(train_file: str):
    """用训练集训练 SentencePiece BPE 模型。"""
    model_file = BPE_MODEL_PREFIX + ".model"
    if os.path.exists(model_file):
        print(f"[✓] BPE 模型已存在: {model_file}")
        return

    print(f"[🔧] 训练 BPE (vocab={VOCAB_SIZE})...")
    print(f"    输入: {train_file}")

    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=BPE_MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        max_sentencepiece_length=16,
        split_digits=True,
    )

    print(f"[✓] BPE 模型: {model_file}")


# ─── 第 2 步：编码 + 分块 ────────────────────────────────────

def encode_text_file(sp, text_file: str) -> torch.Tensor:
    """
    读取文本 → 每行包 BOS + EOS → 全部连接成大序列。
    """
    all_tokens = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            all_tokens.append(BOS_ID)
            all_tokens.extend(ids)
            all_tokens.append(EOS_ID)

    return torch.tensor(all_tokens, dtype=torch.long)


def chunkify(tokens: torch.Tensor, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """切成长度 seq_len 的块，丢弃尾部不足的部分。"""
    n_chunks = len(tokens) // seq_len
    trimmed = tokens[: n_chunks * seq_len]
    return trimmed.reshape(n_chunks, seq_len)


def process_all(sp, files: dict):
    """编码三个文件并分块，保存 .pt。"""
    for split, path in files.items():
        out_path = os.path.join(DATA_DIR, f"{split}.pt")
        if os.path.exists(out_path):
            print(f"[✓] {split}.pt 已存在，跳过。")
            continue

        print(f"[🔢] 编码 {split}...")
        tokens = encode_text_file(sp, path)
        chunks = chunkify(tokens)

        torch.save(chunks, out_path)
        print(f"    {split}: {chunks.shape[0]} 块 × {chunks.shape[1]} tokens")
        print(f"    (原始 {len(tokens)} tokens, 丢弃 {len(tokens) % SEQ_LEN})")


# ─── 第 3 步：加载分词器 ─────────────────────────────────────

def load_sp():
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL_PREFIX + ".model")
    return sp


# ─── 入口 ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  V4 Baseline — 数据准备")
    print("=" * 60)

    files = download_wikitext2()
    train_bpe(files["train"])

    sp = load_sp()
    print(f"\n[📊] 词表大小: {sp.get_piece_size()}")
    print(f"    特殊 token: pad={sp.pad_id()} bos={sp.bos_id()} "
          f"eos={sp.eos_id()} unk={sp.unk_id()}")

    process_all(sp, files)

    print("\n" + "=" * 60)
    print("  数据准备完成！")
    print("=" * 60)
    for split in ["train", "valid", "test"]:
        p = os.path.join(DATA_DIR, f"{split}.pt")
        if os.path.exists(p):
            t = torch.load(p, weights_only=True)
            print(f"  {split}: {t.shape}")
