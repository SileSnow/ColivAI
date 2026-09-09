"""
v3_zero.py — v3 重构版：三步走推理架构
=======================================
核心变更（vs v3_grpo_chain.py）:
  1. Prefix LM 注意力掩码 — prompt 双向，生成部分因果
  2. (tanh, sech²) 相对位置编码，去掉 abs，保留方向信号
  3. Dropout = 0.0（彻底关掉，模型欠拟合不需要正则化）
  4. log_tau 初始化为 ln(3.0)（τ≈3，适合短链注意力范围）
  5. 冷启动数据 500 → 5000+，分训练/验证集
  6. SFT 早停条件从 loss 改为验证集准确率 > 90%
  7. GRPO 去掉 buffer 回放（只对模型自己的生成采样）

设计理念：模型在每步运算中执行「三步走」认知循环——
  Step k: ① Lookback（看上一步结果，offset=-1）
          ② Lookup（读题，覆盖 prompt 区域）
          ③ Compute（计算，offset=0 聚合信息）
  各注意力头通过 w/v/log_tau 自组织分化为抄写头/读题头/计算头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import time
import os
import math
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional

# ============================================================
#  复用基础组件
# ============================================================
import sys
sys.path.insert(0, '/root/airesearch/v2')
from v2 import (
    CharTokenizer, PAD_IDX, BOS_IDX, EOS_IDX,
)


# ============================================================
#  (#1, #2) 注意力层：Prefix LM + 带方向的 tanh/sech² 位置编码
# ============================================================

class SelfAttentionWithCache(nn.Module):
    """
    自注意力层，支持：
    - Prefix LM 掩码：prompt 内双向，生成部分因果
    - (tanh, sech²) 相对位置编码，每头独立 (w, v, τ) 可学习
    - KV cache 加速生成
    - Dropout = 0.0（默认）
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # (#2) (tanh, sech²) 相对位置编码：每头独立参数
        self.w_param = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.v_param = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.3)
        # (#4) τ 初始化为 ln(3.0) ≈ 1.099，τ ≈ 3
        self.log_tau = nn.Parameter(torch.full((n_heads, 1, 1), math.log(3.0)))

    def forward(self, x, cache_k=None, cache_v=None, prompt_length=None):
        """
        x: (B, T, d_model)
        cache_k, cache_v: (B, n_heads, cache_len, d_head) 或 None
        prompt_length: int, prompt 部分的 token 数（用于 Prefix LM 掩码）
                       训练时传入，生成时（有 cache）不需要
        Returns: (output, new_k, new_v)
        """
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if cache_k is not None:
            K = torch.cat([cache_k, K], dim=2)
            V = torch.cat([cache_v, V], dim=2)

        # 注意力分数
        attn = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # (#2) 相对位置偏置：去掉 abs，保留方向信号
        total_len = K.size(2)
        q_pos = torch.arange(total_len - T, total_len, device=x.device)  # (T,)
        k_pos = torch.arange(total_len, device=x.device)                # (total_len,)
        offset = k_pos[None, :] - q_pos[:, None]  # (T, total_len)，带符号！
        tau = torch.exp(self.log_tau) + 0.1        # (n_heads, 1, 1)
        # 不使用 abs，offset 可为负
        # w_param * tanh(-offset/τ): offset>0(右)时为负偏置，offset<0(左)时为正偏置
        # v_param * sech²(offset/τ): sech² 是偶函数，在 offset=0 处最大
        tanh_bias = self.w_param * torch.tanh(-offset / tau) + \
                    self.v_param * (1.0 / torch.cosh(offset / tau))
        tanh_bias = torch.nan_to_num(tanh_bias, nan=0.0)
        attn = attn + tanh_bias.unsqueeze(0)

        # (#1) Prefix LM 掩码 / 因果掩码
        if cache_k is not None:
            # 生成阶段：有 KV cache，每次只生成 1 个 token，不需要额外掩码
            # （cache 中的 past token 已经被掩码过）
            pass
        else:
            # 训练/预填充阶段：全序列一次处理
            if prompt_length is not None:
                if prompt_length >= total_len:
                    # 全部是 prompt，全双向注意力
                    # 不施加掩码，即所有 token 互相关注
                    pass
                else:
                    # (#1) Prefix LM: prompt 内双向，生成部分因果
                    mask = torch.full((T, total_len), float('-inf'), device=x.device)
                    # Prompt 区：所有 prompt token 互相关注（双向）
                    mask[:prompt_length, :prompt_length] = 0.0
                    # 生成区：关注 prompt 全部 + 之前生成的所有 token（含自己）
                    for i in range(prompt_length, T):
                        mask[i, :i+1] = 0.0
                    attn = attn + mask
            else:
                # 标准因果掩码（全序列都是生成部分或无 prompt 划分）
                causal_mask = torch.triu(
                    torch.full((T, T), float('-inf'), device=x.device),
                    diagonal=1
                )
                attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # softmax 后防 NaN
        attn = self.attn_dropout(attn)
        out = attn @ V
        out = torch.nan_to_num(out, nan=0.0)    # 注意力输出防 NaN

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        out = self.dropout(out)

        return out, K, V


class DecoderLayerWithCache(nn.Module):
    """Pre-LN Decoder 层，支持 KV cache + Prefix LM"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = SelfAttentionWithCache(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cache_k=None, cache_v=None, prompt_length=None):
        attn_out, new_k, new_v = self.self_attn(
            self.norm1(x), cache_k, cache_v, prompt_length
        )
        # NaN 防护：注意力输出出现 NaN 时置零
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        ffn_out = torch.nan_to_num(ffn_out, nan=0.0)
        x = x + ffn_out
        return x, new_k, new_v


class TransformerCoTWithCache(nn.Module):
    """
    带 KV cache + Prefix LM 的 Transformer Decoder。
    4 层，4 头，d_model=128，d_ff=512。
    """
    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, d_ff: int = 512,
                 max_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        # 不使用绝对位置编码，相对位置由注意力层中的 (tanh, sech²) 偏置处理
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayerWithCache(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and not name.endswith(('w_param', 'v_param', 'log_tau')):
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_mask=None, past_kv=None, return_kv=False, prompt_length=None):
        """
        x: (batch, seq_len) token ids
        past_kv: list of [(K, V), ...] for each layer
        return_kv: 是否返回 KV cache
        prompt_length: prompt 部分的 token 数（Prefix LM 掩码用）
        """
        seq_len = x.size(1)
        h = self.token_embedding(x) * math.sqrt(self.d_model)
        h = self.dropout(h)

        new_kv = []
        for i, layer in enumerate(self.layers):
            k, v = past_kv[i] if past_kv is not None else (None, None)
            h, new_k, new_v = layer(h, k, v, prompt_length=prompt_length)
            new_kv.append((new_k, new_v))

        h = self.norm(h)
        h = torch.nan_to_num(h, nan=0.0)
        logits = self.output_proj(h)
        logits = torch.clamp(logits, min=-50, max=50)  # 防溢出
        logits = torch.nan_to_num(logits, nan=0.0)     # 防 NaN

        if return_kv:
            return logits, new_kv
        return logits

    # ---------- 生成（KV cache 加速）----------

    @torch.no_grad()
    def generate_with_cache(self, prompt: str, tokenizer: CharTokenizer,
                            max_gen_len: int = 60, temperature: float = 0.8,
                            top_p: float = 0.9) -> Tuple[str, List[int]]:
        """KV cache 加速生成，带 top_p 采样"""
        self.eval()
        device = next(self.parameters()).device

        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_special=True),
            dtype=torch.long
        ).unsqueeze(0).to(device)

        prompt_len = prompt_ids.size(1)

        # (#1) 处理 prompt 并缓存 KV，传入 prompt_length 启用 Prefix LM 掩码
        _, past_kv = self(prompt_ids, return_kv=True, prompt_length=prompt_len)

        current = prompt_ids[:, -1:]
        generated_ids = []

        for _ in range(max_gen_len):
            # 生成阶段有 KV cache，不需要再传 prompt_length
            logits, past_kv = self(current, past_kv=past_kv, return_kv=True)
            next_logits = logits[0, -1, :]

            # top_p 采样
            next_logits = torch.clamp(next_logits, min=-50, max=50)
            scaled = next_logits / temperature
            probs = F.softmax(scaled, dim=-1)

            # Top-P 过滤
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs_clone = sorted_probs.clone()
            sorted_probs_clone[mask] = 0.0

            filtered = torch.zeros_like(probs)
            filtered.scatter_(-1, sorted_idx, sorted_probs_clone)
            probs_sum = filtered.sum()
            if probs_sum > 0:
                probs = filtered / probs_sum
            else:
                probs = torch.ones_like(probs) / probs.size(-1)

            # 禁止 PAD/BOS 生成
            probs[PAD_IDX] = 0.0
            probs[BOS_IDX] = 0.0
            probs = probs / probs.sum()

            token_id = torch.multinomial(probs, 1).item()
            generated_ids.append(token_id)

            if token_id == EOS_IDX:
                break

            current = torch.tensor([[token_id]], dtype=torch.long, device=device)

        text = tokenizer.decode(generated_ids)
        return text, generated_ids

    # ---------- 批量生成：共享 prompt KV cache ----------
    @torch.no_grad()
    def generate_batch(self, prompt: str, tokenizer: CharTokenizer,
                       n_samples: int = 8, max_gen_len: int = 60,
                       temperature: float = 0.8, top_p: float = 0.9) -> List[Tuple[str, List[int]]]:
        """从同一 prompt 生成多个样本，共享 prompt KV cache"""
        self.eval()
        device = next(self.parameters()).device

        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_special=True),
            dtype=torch.long
        ).unsqueeze(0).to(device)

        prompt_len = prompt_ids.size(1)

        # 处理 prompt 一次，缓存 KV（传入 prompt_length）
        _, base_kv = self(prompt_ids, return_kv=True, prompt_length=prompt_len)
        last_token = prompt_ids[:, -1:]

        results = []
        for _ in range(n_samples):
            past_kv = base_kv
            current = last_token
            generated_ids = []

            for _ in range(max_gen_len):
                logits, past_kv = self(current, past_kv=past_kv, return_kv=True)
                next_logits = logits[0, -1, :]
                next_logits = torch.clamp(next_logits, min=-50, max=50)
                scaled = next_logits / temperature
                probs = F.softmax(scaled, dim=-1)

                # Top-P 过滤
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs_clone = sorted_probs.clone()
                sorted_probs_clone[mask] = 0.0

                filtered = torch.zeros_like(probs)
                filtered.scatter_(-1, sorted_idx, sorted_probs_clone)
                probs_sum = filtered.sum()
                if probs_sum > 0:
                    probs = filtered / probs_sum
                else:
                    probs = torch.ones_like(probs) / probs.size(-1)

                # 禁止 PAD/BOS 生成
                probs[PAD_IDX] = 0.0
                probs[BOS_IDX] = 0.0
                probs = probs / probs.sum()

                token_id = torch.multinomial(probs, 1).item()
                generated_ids.append(token_id)

                if token_id == EOS_IDX:
                    break

                current = torch.tensor([[token_id]], dtype=torch.long, device=device)

            text = tokenizer.decode(generated_ids)
            results.append((text, generated_ids))

        return results


# ============================================================
#  连续链数据生成（#5 扩充到 5000+）
# ============================================================

def generate_chain_cot(start: int, ops: List[str], nums: List[int]) -> str:
    """
    生成连续链 CoT。
    例：start=12, ops=['-','+'], nums=[3,5] →
      12-3+5=
      12-3=9
      9+5=14
      =14
    """
    lines = []
    expr = str(start)
    for op, num in zip(ops, nums):
        expr += f"{op}{num}"
    expr += "="
    lines.append(expr)

    current = start
    for op, num in zip(ops, nums):
        if op == '+':
            result = current + num
        else:
            result = current - num
        lines.append(f"{current}{op}{num}={result}")
        current = result
    lines.append(f"={current}")
    return "\n".join(lines)


def generate_cold_start_data(n_unique: int = 150, repeats: int = 5,
                              min_steps: int = 2, max_steps: int = 5,
                              max_val: int = 50, seed: int = 42,
                              val_ratio: float = 0.1):
    """
    生成冷启动 CoT 数据，带可控重复，适合小模型学习。
    
    - n_unique: 唯一 (start, ops) 组合数
    - repeats: 每个组合的重复次数（不同操作数值）
    - 总样本数 = n_unique x repeats，每个组合的 ops 固定，nums 随机

    返回:
      train_data: List[(prompt_text, gt, prompt_tokens, cot_tokens, prompt_len)]
      val_data:   List[(prompt_text, gt, prompt_tokens, cot_tokens, prompt_len)]
    """
    tokenizer = CharTokenizer()
    rng = np.random.RandomState(seed)

    all_data = []

    for pair_idx in range(n_unique):
        n_steps = rng.randint(min_steps, max_steps + 1)
        start = int(rng.randint(-max_val, max_val + 1))
        ops = [rng.choice(["+" , "-"]) for _ in range(n_steps)]

        for rep in range(repeats):
            nums = [int(rng.randint(0, max_val + 1)) for _ in range(n_steps)]

            cot = generate_chain_cot(start, ops, nums)
            lines_list = cot.split("\n")
            prompt = lines_list[0]
            gt_line = lines_list[-1]
            gt = int(gt_line[1:])

            prompt_tokens = tokenizer.encode(prompt, add_special=True)
            cot_tokens = tokenizer.encode(cot, add_special=True)
            prompt_len = len(prompt_tokens) - 1

            all_data.append((prompt, gt, prompt_tokens, cot_tokens, prompt_len))

    rng.shuffle(all_data)

    split_idx = int(len(all_data) * (1 - val_ratio))
    train_data = [(p, g, pt, ct, pl) for p, g, pt, ct, pl in all_data[:split_idx]]
    val_data = [(p, g, pt, ct, pl) for p, g, pt, ct, pl in all_data[split_idx:]]

    n_total = len(all_data)
    print(f"  [ColdStart] {n_unique} 种子 x {repeats} 重复 = {n_total} 条, "
          f"训练 {len(train_data)} + 验证 {len(val_data)}")
    return train_data, val_data
def compute_step_reward(response: str, ground_truth: int, prompt: str = "") -> float:
    """
    连续链 CoT 奖励 v2 — reward ×2 + 稠密格式奖励。

    逐步骤奖励（×2）：
      ✅ 完美 step (A✓ B✓ C✓) → +1.0
      🟡 A✓ B✓ 但 C 算错      → +0.2
      🟠 A✓ 但 B 错（抄错题） → -0.4
      🔴 A 错（链断了）        → -0.6
    bonus: 最终答案正确 → +0.4
    无任何 CoT 步骤 → -1.0

    格式奖励（每行）：
      ⭐ 有数字+运算符+=号 → +0.1
      ⭐ 有数字和=号       → +0.05
      ⭐ 有数字            → +0.02
      格式奖励封顶 +0.5
    """
    lines = response.strip().split('\n')
    reward = 0.0
    step_re = re.compile(r'^(-?\d+)([+-])(-?\d+)=(-?\d+)$')

    # 从 prompt 解析操作数序列和运算符序列
    prompt_nums = []
    prompt_ops = []
    if prompt:
        p = prompt.rstrip('=')
        parts = re.findall(r'[+-]?\d+', p)
        if parts:
            prompt_nums.append(int(parts[0]))
            for part in parts[1:]:
                if part[0] in '+-':
                    prompt_ops.append(part[0])
                    prompt_nums.append(int(part[1:]))

    prev_c = None
    n_steps = 0
    step_idx = 0

    # ---------- 逐步骤奖励（×2） ----------
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = step_re.match(line)
        if m:
            a = int(m.group(1))
            op = m.group(2)
            b = int(m.group(3))
            c = int(m.group(4))
            n_steps += 1

            # 左目校验：A 是否等于上一步结果
            a_correct = (prev_c is None) or (a == prev_c)

            # 右目校验：B 是否等于题目中对应的操作数
            b_correct = True
            if step_idx < len(prompt_ops) and step_idx < len(prompt_nums) - 1:
                expected_b = prompt_nums[step_idx + 1]
                b_correct = (b == expected_b)
            elif prev_c is None:
                if prompt_nums:
                    a_correct = a_correct and (a == prompt_nums[0])

            # 校验 C
            if op == '+':
                c_correct = (a + b == c)
            else:
                c_correct = (a - b == c)

            if a_correct and b_correct and c_correct:
                reward += 1.0
            elif a_correct and b_correct and not c_correct:
                reward += 0.2
            elif a_correct and not b_correct:
                reward -= 0.4
            else:
                reward -= 0.6

            prev_c = c
            step_idx += 1

        elif line.startswith('='):
            try:
                final = int(line[1:].strip())
                if final == ground_truth:
                    reward += 0.4
            except ValueError:
                pass

    if n_steps == 0:
        reward -= 1.0

    # ---------- 稠密格式奖励（每行给分，不重叠） ----------
    format_bonus = 0.0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 只评估「格式贴近度」——不依赖数学正确性
        has_digit = bool(re.search(r'\d', line))
        has_op = bool(re.search(r'[+-]', line))
        has_eq = '=' in line

        if has_digit and has_op and has_eq:
            format_bonus += 0.1
        elif has_digit and has_eq:
            format_bonus += 0.05
        elif has_digit:
            format_bonus += 0.02

    format_bonus = min(format_bonus, 0.5)
    reward += format_bonus

    return reward


# ============================================================
#  可微 Log Prob 计算
# ============================================================

def compute_log_probs(model, prompt_tokens: List[int], response_tokens: List[int],
                      device: str = 'cpu') -> torch.Tensor:
    """计算 response 部分每个 token 的 log 概率"""
    full_ids = prompt_tokens + response_tokens
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_tokens)

    logits = model(full_tensor)
    resp_logits = logits[0, prompt_len - 1: -1, :]
    resp_logits = torch.clamp(resp_logits, min=-50, max=50)
    log_prob_dist = F.log_softmax(resp_logits, dim=-1)
    if torch.isnan(log_prob_dist).any():
        log_prob_dist = torch.nan_to_num(log_prob_dist, nan=-100.0)

    resp_tensor = torch.tensor(response_tokens, dtype=torch.long, device=resp_logits.device)
    gathered = log_prob_dist[torch.arange(len(response_tokens), device=resp_logits.device), resp_tensor]
    gathered = torch.clamp(gathered, min=-100, max=0)

    return gathered


# ============================================================
#  SFT 热身（#5, #6: 5000+ 数据 + 验证集准确率早停）
# ============================================================

def evaluate_sft_accuracy(model, tokenizer, val_data, device,
                          max_gen_len=60, num_samples=100):
    """
    在验证集上评估模型的生成准确率。
    val_data: List[(prompt, gt, prompt_tokens, cot_tokens, prompt_len)]
    返回: accuracy (float, 0~1)
    """
    model.eval()
    correct = 0
    total = min(num_samples, len(val_data))

    for i in range(total):
        prompt_text, gt, _, _, _ = val_data[i]
        resp, _ = model.generate_with_cache(
            prompt_text, tokenizer, max_gen_len=max_gen_len,
            temperature=0.6, top_p=0.9
        )
        m = re.search(r'=(-?\d+)\s*$', resp)
        if m and int(m.group(1)) == gt:
            correct += 1

    model.train()
    return correct / max(total, 1)


def run_sft(model, tokenizer, train_data, val_data, device,
            sft_epochs=200, batch_size=32, lr=1e-3,
            val_interval=5, target_acc=0.9, early_stop_patience=10):
    """
    SFT 热身训练。
    早停条件：验证集准确率 > target_acc 或 连续 early_stop_patience 轮验证无改善。
    """
    print(f"\n=== SFT 热身：{len(train_data)} 训练 + {len(val_data)} 验证 ===\n"
          f"    早停目标：验证集准确率 > {target_acc*100:.0f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0.0
    no_improve = 0
    best_model_state = None

    # 准备训练数据（只取 cot_tokens）
    train_seqs = [ct for _, _, _, ct, _ in train_data]
    val_seqs = [ct for _, _, _, ct, _ in val_data]

    for sft_epoch in range(sft_epochs):
        random.shuffle(train_seqs)
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_seqs), batch_size):
            batch = train_seqs[batch_start:batch_start + batch_size]
            max_len_batch = max(len(seq) for seq in batch)
            padded = torch.full((len(batch), max_len_batch), PAD_IDX,
                                dtype=torch.long, device=device)
            for i, seq in enumerate(batch):
                padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

            # SFT 使用因果掩码（不传 prompt_length），Prefix LM 在生成/RL 阶段启用
            logits = model(padded)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = padded[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=PAD_IDX
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # 定期验证
        do_val = ((sft_epoch + 1) % val_interval == 0) or (sft_epoch == 0)
        val_acc = 0.0
        if do_val:
            val_acc = evaluate_sft_accuracy(
                model, tokenizer, val_data, device,
                num_samples=min(100, len(val_data))
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                no_improve = 0
                print(f"  SFT Epoch {sft_epoch+1:>3}  Loss: {avg_loss:.4f}  "
                      f"Val Acc: {val_acc*100:.1f}%  🏆 新最佳")
            else:
                no_improve += 1
                print(f"  SFT Epoch {sft_epoch+1:>3}  Loss: {avg_loss:.4f}  "
                      f"Val Acc: {val_acc*100:.1f}%  (最佳 {best_val_acc*100:.1f}%)")
        else:
            # 简略输出
            if (sft_epoch + 1) % 10 == 0:
                print(f"  SFT Epoch {sft_epoch+1:>3}  Loss: {avg_loss:.4f}")

        # 早停检查
        if best_val_acc >= target_acc:
            print(f"  ✅ 验证集准确率 {best_val_acc*100:.1f}% > 目标 {target_acc*100:.0f}%，SFT 结束")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            return best_val_acc

        if no_improve >= early_stop_patience and sft_epoch >= 20:
            print(f"  ↪ 早停于 Epoch {sft_epoch+1}（连续 {early_stop_patience} 轮验证无改善）")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            return best_val_acc

    print(f"  ✅ SFT 完成（{sft_epochs} epoch），最佳 Val Acc: {best_val_acc*100:.1f}%")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return best_val_acc


# ============================================================
#  GRPO 训练（#7: 去回放，只对模型自己生成采样）
# ============================================================

def train_grpo_epoch(model, tokenizer, dataset, optimizer, device,
                     num_samples: int = 8, temperature: float = 0.8,
                     top_p: float = 0.9, max_gen_len: int = 60,
                     max_prompts: int = 500,
                     ref_model=None, kl_beta: float = 0.05) -> Tuple[float, float, float, float]:
    """
    GRPO 训练一个 epoch。
    - 不再使用 replay buffer，所有样本均为模型当前生成
    - ref_model: SFT 后的冻结模型，用于 KL 惩罚
    """
    model.eval()
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:max_prompts]

    total_rewards = []
    total_policy_loss = 0.0
    total_kl = 0.0
    n_kl_batches = 0
    correct_count = 0
    total_count = 0

    for batch_idx, idx in enumerate(indices):
        prompt, gt, prompt_tokens = dataset[idx]

        # ---- 全部生成，不回放 ----
        responses = []
        all_resp_tokens = []
        response_prompts = []
        response_prompt_tokens = []
        response_gts = []

        # 生成 n_samples 个回答（共享 prompt KV cache）
        batch_results = model.generate_batch(
            prompt, tokenizer,
            n_samples=num_samples,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )
        for resp_text, resp_tokens in batch_results:
            responses.append(resp_text)
            all_resp_tokens.append(resp_tokens)
            response_prompts.append(prompt)
            response_prompt_tokens.append(prompt_tokens)
            response_gts.append(gt)

        # ---- 计算奖励 ----
        rewards = [compute_step_reward(resp, response_gts[i], response_prompts[i])
                   for i, resp in enumerate(responses)]

        for i, resp in enumerate(responses):
            m = re.search(r'=(-?\d+)', resp)
            if m and int(m.group(1)) == response_gts[i]:
                correct_count += 1
            total_count += 1

        # ---- 组内 advantage ----
        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        # ---- 策略梯度 + KL 惩罚 ----
        policy_loss = torch.tensor(0.0, device=device)
        kl_loss_sum = torch.tensor(0.0, device=device)
        n_kl_valid = 0

        for i in range(len(all_resp_tokens)):
            if len(all_resp_tokens[i]) == 0:
                continue
            p_tokens_i = response_prompt_tokens[i]
            log_probs = compute_log_probs(model, p_tokens_i, all_resp_tokens[i], device)
            policy_loss = policy_loss - advantages[i] * log_probs.sum()

            if ref_model is not None and kl_beta > 0:
                with torch.no_grad():
                    ref_log_probs = compute_log_probs(ref_model, p_tokens_i, all_resp_tokens[i], device)
                d = ref_log_probs - log_probs
                d_clamped = torch.clamp(d, min=-20, max=20)
                kl_per_token = torch.exp(d_clamped) - d - 1
                kl_per_token = torch.clamp(kl_per_token, max=5.0)
                if torch.isnan(kl_per_token).any():
                    kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0)
                kl_div = kl_per_token.mean()
                kl_loss_sum = kl_loss_sum + kl_div
                n_kl_valid += 1

        policy_loss = policy_loss / num_samples
        if ref_model is not None and kl_beta > 0 and n_kl_valid > 0:
            kl_mean = kl_loss_sum / n_kl_valid
            kl_penalty = kl_beta * kl_mean
            policy_loss = policy_loss + kl_penalty

        optimizer.zero_grad()
        if torch.isnan(policy_loss).any():
            policy_loss = torch.tensor(0.0, device=device)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        if ref_model is not None and kl_beta > 0 and n_kl_valid > 0:
            total_kl += (kl_loss_sum / n_kl_valid).item()
            n_kl_batches += 1
        total_rewards.append(mean_r.item())

        if (batch_idx + 1) % 100 == 0:
            avg_r = np.mean(total_rewards[-100:]) if total_rewards else 0
            ts = time.strftime("%H:%M:%S")
            print(f"    [{ts}] [{batch_idx+1}/{len(indices)}]  "
                  f"Reward: {avg_r:.3f}  Loss: {policy_loss.item():.4f}", flush=True)

    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    avg_loss = total_policy_loss / max(len(indices), 1)
    accuracy = correct_count / max(total_count, 1) * 100
    avg_kl = total_kl / max(n_kl_batches, 1) if ref_model is not None and kl_beta > 0 else 0.0

    return avg_reward, avg_loss, accuracy, avg_kl


# ============================================================
#  Checkpoint 管理器
# ============================================================

class CheckpointManager:
    def __init__(self, save_dir: str, model_name: str = "transformer_cot_v3_zero"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_reward = -float('inf')
        self.best_epoch = -1

    def save(self, model, optimizer, epoch, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_reward': self.best_reward,
            'best_epoch': self.best_epoch,
            'metrics': metrics,
        }
        latest_path = self.save_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        if epoch % 5 == 0:
            path = self.save_dir / f"{self.model_name}_epoch{epoch}.pth"
            torch.save(checkpoint, path)
            print(f"    [Checkpoint] 已保存: {path.name}")

        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"    [Checkpoint] 🏆 新最佳! Reward: {metrics.get('avg_reward', 0):.4f}")

    def load_latest(self, model, optimizer=None) -> Optional[int]:
        latest_path = self.save_dir / f"{self.model_name}_latest.pth"
        if latest_path.exists():
            ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.best_reward = ckpt.get('best_reward', -float('inf'))
            epoch = ckpt['epoch']
            print(f"  [Checkpoint] 恢复于 Epoch {epoch + 1}")
            return epoch
        return None


# ============================================================
#  RL 数据集（和之前一致，但基于新生成的数据）
# ============================================================

class ChainDataset(Dataset):
    """连续链 CoT 数据集。和之前一致。"""
    def __init__(self, num_samples: int = 12000,
                 min_steps: int = 2, max_steps: int = 8,
                 max_val: int = 200, seed: int = 123):
        self.tokenizer = CharTokenizer()
        rng = np.random.RandomState(seed)

        self.prompts: List[str] = []
        self.ground_truths: List[int] = []
        self.prompt_tokens: List[List[int]] = []
        self.cots: List[str] = []

        for _ in range(num_samples):
            n_steps = rng.randint(min_steps, max_steps + 1)
            start = int(rng.randint(-50, max_val + 1))
            ops = []
            nums = []
            for _ in range(n_steps):
                if rng.random() < 0.5:
                    ops.append('+')
                else:
                    ops.append('-')
                nums.append(int(rng.randint(0, max_val + 1)))

            cot = generate_chain_cot(start, ops, nums)
            lines = cot.split('\n')
            prompt = lines[0]
            gt_line = lines[-1]
            gt = int(gt_line[1:])

            self.prompts.append(prompt)
            self.ground_truths.append(gt)
            self.prompt_tokens.append(
                self.tokenizer.encode(prompt, add_special=True))
            self.cots.append(cot)

        combined = list(zip(self.prompts, self.ground_truths,
                           self.prompt_tokens, self.cots))
        rng.shuffle(combined)
        self.prompts = [c[0] for c in combined]
        self.ground_truths = [c[1] for c in combined]
        self.prompt_tokens = [c[2] for c in combined]
        self.cots = [c[3] for c in combined]

        print(f"  [ChainDataset] {len(self.prompts)} 条连续链, "
              f"{min_steps}~{max_steps} 步 ✅")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.ground_truths[idx], self.prompt_tokens[idx]


# ============================================================
#  主训练
# ============================================================

def main():
    # ======== 配置 ========
    D_MODEL     = 128
    N_LAYERS    = 4       # (#2.4) 保持 4 层——三步走 + 进位需要
    N_HEADS     = 4
    D_FF        = 512
    MAX_LEN     = 256
    DROPOUT     = 0.0     # (#3) 彻底关掉 Dropout

    # === 冷启动 / SFT ===
    COLD_N_UNIQUE = 150      # 唯一 (start, ops) 组合数
    COLD_REPEATS = 5         # 每个组合重复次数（不同操作数）
    # 总样本 = 150 x 5 = 750，其中训练 675 + 验证 75
    SFT_MIN_EPOCHS = 20
    SFT_MAX_EPOCHS = 200
    SFT_LR = 1e-3
    SFT_BATCH_SIZE = 32
    SFT_VAL_INTERVAL = 5      # 每 5 epoch 验证一次
    SFT_TARGET_ACC = 0.9      # (#6) 验证集准确率 > 90%
    SFT_EARLY_STOP = 10       # 连续 10 轮无改善则早停

    # === RL / GRPO ===
    RL_EPOCHS       = 30
    RL_LR           = 3e-5
    RL_KL_BETA      = 0.01
    RL_TEMPERATURE  = 0.2
    RL_TOP_P        = 0.9
    RL_MAX_GEN_LEN  = 40
    RL_NUM_SAMPLES  = 8
    RL_PROMPTS_PER_EPOCH = 500

    NUM_SAMPLES     = 12000   # RL 阶段用的 eval 数据集
    MAX_VAL         = 200

    CKPT_DIR = "/root/airesearch/v3/checkpoints_v3_zero"
    LOG_PATH = "/root/airesearch/v3/train_v3_zero.log"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")
    print(f"存档: {CKPT_DIR}")
    print(f"日志: {LOG_PATH}")

    # ======== 模型 ========
    print("\n=== 构建模型… ===")
    tokenizer = CharTokenizer()
    model = TransformerCoTWithCache(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF,
        max_len=MAX_LEN, dropout=DROPOUT
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")
    print(f"  Dropout: {DROPOUT}  |  "
          f"τ 初始化: ln(3.0) ≈ 1.099  |  "
          f"位置编码: (tanh, sech²) 无 abs")

    # ======== 冷启动数据生成 + SFT 热身 ========
    ckpt_manager = CheckpointManager(CKPT_DIR)

    train_data, val_data = generate_cold_start_data(
        n_unique=COLD_N_UNIQUE, repeats=COLD_REPEATS,
        min_steps=2, max_steps=5,
        max_val=50, seed=42,
        val_ratio=0.1
    )

    # 检查 SFT checkpoint，有则跳过 SFT（避免重启后重新跑 45 epoch）
    sft_ckpt = f"{CKPT_DIR}/sft_best.pth"
    if os.path.exists(sft_ckpt):
        model.load_state_dict(torch.load(sft_ckpt, map_location=DEVICE))
        print(f"  ✅ 从 {sft_ckpt} 恢复 SFT 模型，跳过 SFT 热身")
        sft_val_acc = 0.0
    else:
        sft_val_acc = run_sft(
            model, tokenizer, train_data, val_data, DEVICE,
            sft_epochs=SFT_MAX_EPOCHS,
            batch_size=SFT_BATCH_SIZE,
            lr=SFT_LR,
            val_interval=SFT_VAL_INTERVAL,
            target_acc=SFT_TARGET_ACC,
            early_stop_patience=SFT_EARLY_STOP
        )
        torch.save(model.state_dict(), sft_ckpt)
        print(f"  💾 SFT 模型已保存至 {sft_ckpt}")
    print(f"  ✅ SFT 热身完成，最终验证集准确率: {sft_val_acc*100:.1f}%")

    # ======== 参考模型（用于 KL 惩罚）=======
    ref_model = None
    if RL_KL_BETA > 0:
        print("\n=== 创建参考模型（深拷贝 SFT 后模型）用于 KL 惩罚… ===")
        ref_model = TransformerCoTWithCache(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL, n_layers=N_LAYERS,
            n_heads=N_HEADS, d_ff=D_FF,
            max_len=MAX_LEN, dropout=DROPOUT
        )
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print("  ✅ 参考模型已冻结（SFT 锚点）")

    # ======== GRPO 优化器 ========
    optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RL_EPOCHS
    )

    # ======== RL 数据集 ========
    print("\n=== 生成连续链数据集… ===")
    rl_dataset = ChainDataset(
        num_samples=NUM_SAMPLES,
        min_steps=1, max_steps=1,
        max_val=MAX_VAL, seed=123
    )

    # ======== GRPO Fine-tune ========
    print(f"\n{'='*60}")
    print(f"🎯 v3-zero GRPO 训练")
    print(f"   采样: T={RL_TEMPERATURE}, top_p={RL_TOP_P}, max_len={RL_MAX_GEN_LEN}")
    print(f"   每组 {RL_NUM_SAMPLES} 样本 | KL β={RL_KL_BETA} | {RL_EPOCHS} epoch")
    print(f"   仅 1 步运算 | 从 SFT 锚点起步")
    print(f"   ❌ 无回放 | ✅ 仅用模型自生成样本")
    print(f"   Dropout={DROPOUT} | Prefix LM ✅ | 位置编码无abs ✅")
    print(f"{'='*60}")

    for rl_epoch in range(RL_EPOCHS):
        epoch_start = time.time()

        avg_reward, avg_loss, accuracy, avg_kl = train_grpo_epoch(
            model, tokenizer, rl_dataset, optimizer, DEVICE,
            num_samples=RL_NUM_SAMPLES,
            temperature=RL_TEMPERATURE,
            top_p=RL_TOP_P,
            max_gen_len=RL_MAX_GEN_LEN,
            max_prompts=RL_PROMPTS_PER_EPOCH,
            ref_model=ref_model, kl_beta=RL_KL_BETA
        )

        scheduler.step()
        elapsed = time.time() - epoch_start

        is_best = avg_reward > ckpt_manager.best_reward
        if is_best:
            ckpt_manager.best_reward = avg_reward
            ckpt_manager.best_epoch = rl_epoch

        metrics = {
            'avg_reward': avg_reward,
            'rl_loss': avg_loss,
            'accuracy': accuracy,
            'phase': 'rl',
            'epoch': rl_epoch,
        }
        ckpt_manager.save(model, optimizer, rl_epoch, metrics, is_best=is_best)

        kl_str = f"  |  KL: {avg_kl:.4f}" if ref_model else ""
        print(f"  RL Epoch {rl_epoch+1}/{RL_EPOCHS}  |  "
              f"Reward: {avg_reward:.4f}  |  Loss: {avg_loss:.4f}  |  "
              f"Acc: {accuracy:.1f}%{kl_str}  |  {elapsed:.1f}s", flush=True)

        # 每个 epoch 打印 2 条生成样本（看看模型在说什么胡话）
        model.eval()
        print("  ── 样本输出 ──", flush=True)
        for _ in range(2):
            idx = random.randint(0, len(rl_dataset) - 1)
            p, gt, _ = rl_dataset[idx]
            resp, _ = model.generate_with_cache(
                p, tokenizer, max_gen_len=RL_MAX_GEN_LEN,
                temperature=0.6, top_p=RL_TOP_P)
            rew = compute_step_reward(resp, gt, p)
            m = re.search(r'=(-?\d+)\s*$', resp)
            pred = m.group(1) if m else "?"
            ok = "✅" if rew >= 0.3 else " "
            resp_show = resp.replace('\n', ' | ')
            print(f"  {ok} P: {p}", flush=True)
            print(f"    → {resp_show}", flush=True)
            print(f"    pred={pred}  gt={gt}  R={rew:.2f}", flush=True)
        model.train()

        # 定期评估
        if (rl_epoch + 1) % 5 == 0:
            print("\n  " + "-" * 50)
            print("  📊 中间评估（采样 10 条）:")
            model.eval()
            correct = 0
            for _ in range(10):
                idx = random.randint(0, len(rl_dataset) - 1)
                p, gt, _ = rl_dataset[idx]
                resp, _ = model.generate_with_cache(
                    p, tokenizer, max_gen_len=RL_MAX_GEN_LEN,
                    temperature=0.6, top_p=RL_TOP_P)
                rew = compute_step_reward(resp, gt, p)
                m = re.search(r'=(-?\d+)\s*$', resp)
                pred = m.group(1) if m else "?"
                ok = "✅" if rew >= 0.3 else " "
                print(f"  {ok} {p} pred={pred} gt={gt} R={rew:.2f}")
                if rew >= 0.3:
                    correct += 1
            model.train()
            print(f"  → 粗略准确率: {correct}/10")
            print("  " + "-" * 50 + "\n")

    # ======== 最终保存 ========
    final_path = f"{CKPT_DIR}/v3_zero_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 最终模型: {final_path}")

    print("\n" + "=" * 60)
    print("🏁 最终评估（20 条连续链）")
    print("=" * 60)
    model.eval()
    correct = 0
    total_reward = 0.0
    for _ in range(20):
        idx = random.randint(0, len(rl_dataset) - 1)
        p, gt, _ = rl_dataset[idx]
        resp, _ = model.generate_with_cache(
            p, tokenizer, max_gen_len=RL_MAX_GEN_LEN,
            temperature=0.6, top_p=RL_TOP_P)
        rew = compute_step_reward(resp, gt, p)
        m = re.search(r'=(-?\d+)\s*$', resp)
        pred = m.group(1) if m else "?"
        ok = "✅" if rew >= 0.3 else " "
        total_reward += rew
        print(f"  {ok} {p:>12} pred={pred:>6} gt={gt:>5} R={rew:+.2f}")
        if rew >= 0.3:
            correct += 1
    print(f"  → 粗略准确率: {correct}/20  平均奖励: {total_reward/20:.3f}")

    print("\n✅ v3-zero 训练完成！")


if __name__ == "__main__":
    print("日志将写入: /root/airesearch/v3/train_v3_zero.log")
    main()
