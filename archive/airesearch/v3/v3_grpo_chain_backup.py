"""
v3_grpo_chain.py — 连续链 + GRPO + tanh 相对位置编码
=====================================================
#1 ✅ 连续链 CoT：可变长度 2~8 步 ± 运算
#2 ✅ tanh 相对位置编码：每注意力头独立 (w, v, τ) 自学习
#3 ✅ 步骤级奖励：三级 + 双目校验
#4 ✅ GRPO 训练：组内 8 采样 + 无偏 KL
#5 ✅ KV Cache 加速
#6 ✅ SFT 热身：冷启动数据 teacher forcing 预训练
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
#  #2 带 KV Cache 的自注意力 & Decoder 层
# ============================================================

class CausalSelfAttentionWithCache(nn.Module):
    """
    因果自注意力，支持 KV cache + tanh 相对位置编码。
    每头独立学习 (w, v, τ) 参数，自组织为计算头/读题头/格式头。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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

        # #2 tanh 相对位置编码：每头独立参数
        self.w_param = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.v_param = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.3)
        self.log_tau = nn.Parameter(torch.zeros(n_heads, 1, 1))

    def forward(self, x, cache_k=None, cache_v=None):
        """
        x: (B, T, d_model)
        cache_k, cache_v: (B, n_heads, cache_len, d_head) 或 None
        Returns: (output, new_k, new_v)
        """
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if cache_k is not None:
            K = torch.cat([cache_k, K], dim=2)
            V = torch.cat([cache_v, V], dim=2)

        attn = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # #2 tanh 相对位置偏置
        total_len = K.size(2)
        q_pos = torch.arange(total_len - T, total_len, device=x.device)  # 查询位置
        k_pos = torch.arange(total_len, device=x.device)                # 键位置
        offset = k_pos[None, :] - q_pos[:, None]  # (T, total_len)
        dist = torch.abs(offset).float()
        tau = torch.exp(self.log_tau) + 0.1        # (n_heads, 1, 1)
        tanh_bias = self.w_param * torch.tanh(-dist / tau) + \
                    self.v_param * (1.0 / torch.cosh(dist / tau))
        # tanh_bias: (n_heads, T, total_len) → broadcast 到 (B, n_heads, T, total_len)
        attn = attn + tanh_bias.unsqueeze(0)

        # 因果掩码
        if cache_k is not None:
            causal_mask = torch.triu(
                torch.full((T, total_len), float('-inf'), device=x.device),
                diagonal=1 + total_len - T
            )
        else:
            causal_mask = torch.triu(
                torch.full((T, T), float('-inf'), device=x.device),
                diagonal=1
            )
        attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        out = self.dropout(out)

        return out, K, V


class DecoderLayerWithCache(nn.Module):
    """Pre-LN Decoder 层，支持 KV cache"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CausalSelfAttentionWithCache(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cache_k=None, cache_v=None):
        attn_out, new_k, new_v = self.self_attn(self.norm1(x), cache_k, cache_v)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_k, new_v


class TransformerCoTWithCache(nn.Module):
    """
    带 KV cache 的 Transformer Decoder。
    参数命名经过设计，可从 v2.py 的 TransformerCoT 转换权重。
    """
    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, d_ff: int = 512,
                 max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        # 不使用 sin/cos 绝对位置编码，改用每头自学习的 tanh 相对偏置（见 CausalSelfAttentionWithCache）
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

    def forward(self, x, src_mask=None, past_kv=None, return_kv=False):
        """
        x: (batch, seq_len) token ids
        past_kv: list of [(K, V), ...] for each layer
        return_kv: 是否返回 KV cache
        """
        seq_len = x.size(1)
        h = self.token_embedding(x) * math.sqrt(self.d_model)
        # 无绝对位置编码，相对位置由每头自学习的 tanh 偏置处理
        h = self.dropout(h)

        new_kv = []
        for i, layer in enumerate(self.layers):
            k, v = past_kv[i] if past_kv is not None else (None, None)
            h, new_k, new_v = layer(h, k, v)
            new_kv.append((new_k, new_v))

        h = self.norm(h)
        logits = self.output_proj(h)
        logits = torch.clamp(logits, min=-50, max=50)  # 防溢出

        if return_kv:
            return logits, new_kv
        return logits

    # ---------- 生成 ----------

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

        # 处理 prompt，缓存 K,V
        _, past_kv = self(prompt_ids, return_kv=True)
        current = prompt_ids[:, -1:]
        generated_ids = []

        for _ in range(max_gen_len):
            logits, past_kv = self(current, past_kv=past_kv, return_kv=True)
            next_logits = logits[0, -1, :]

            # top_p 采样（防溢出）
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
                probs = torch.ones_like(probs) / probs.size(-1)  # fallback: 均匀分布

            token_id = torch.multinomial(probs, 1).item()
            generated_ids.append(token_id)

            if token_id == EOS_IDX:
                break

            current = torch.tensor([[token_id]], dtype=torch.long, device=device)

        text = tokenizer.decode(generated_ids)
        return text, generated_ids

    # ---------- 兼容原 generate 接口 ----------
    @torch.no_grad()
    def generate(self, prompt: str, tokenizer: CharTokenizer,
                 max_gen_len: int = 200, temperature: float = 0.0) -> str:
        text, _ = self.generate_with_cache(prompt, tokenizer, max_gen_len, max(0.001, temperature) if temperature > 0 else 0.001, top_p=1.0)
        return text


# ============================================================
#  ⚡ 权重转换：TransformerCoT → TransformerCoTWithCache
# ============================================================

def convert_sft_checkpoint(old_ckpt_path: str, new_model: TransformerCoTWithCache,
                           device: str = 'cpu') -> None:
    """
    将 v2.py SFT 训练的 TransformerCoT 权重加载到 TransformerCoTWithCache。
    
    命名映射:
      TransformerCoT (nn.TransformerDecoderLayer)
        layers.i.self_attn.in_proj_weight  →  W_q.weight, W_k.weight, W_v.weight
        layers.i.self_attn.in_proj_bias     →  W_q.bias, W_k.bias, W_v.bias
        layers.i.self_attn.out_proj.weight  →  W_o.weight
        layers.i.self_attn.out_proj.bias    →  W_o.bias
        layers.i.norm1.*                    →  norm1.* (相同)
        layers.i.linear1.*                  →  ffn.0.*
        layers.i.linear2.*                  →  ffn.3.*
        layers.i.norm2.*                    →  norm2.* (相同)
        token_embedding.*                   →  token_embedding.* (相同)
        pos_encoding.pe                     →  pos_encoding.pe (相同)
        norm.*                              →  norm.* (相同)
        output_proj.*                       →  output_proj.* (相同)
    """
    print(f"  加载 SFT checkpoint: {old_ckpt_path}")

    # 加载 checkpoint（可能是完整存档或纯 state_dict）
    raw = torch.load(old_ckpt_path, map_location=device)
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        old_sd = raw['model_state_dict']
    elif isinstance(raw, dict) and 'state_dict' in raw:
        old_sd = raw['state_dict']
    else:
        old_sd = raw

    new_sd = {}

    # 直接复制相同名字的参数
    direct_copy = [
        'token_embedding.weight',
        'norm.weight', 'norm.bias',
        'output_proj.weight', 'output_proj.bias',
        'pos_encoding.pe',
    ]
    for name in direct_copy:
        if name in old_sd:
            new_sd[name] = old_sd[name]

    # 逐层转换
    n_layers = new_model.n_layers
    d_model = new_model.d_model

    for i in range(n_layers):
        # 1. Self-Attention: 拆分 in_proj_weight
        in_proj_w = old_sd.get(f'layers.{i}.self_attn.in_proj_weight')
        if in_proj_w is not None:
            # nn.MultiheadAttention 的 in_proj 是 Q,K,V 拼接: shape (3*d_model, d_model)
            q_w, k_w, v_w = in_proj_w.chunk(3, dim=0)
            new_sd[f'layers.{i}.self_attn.W_q.weight'] = q_w
            new_sd[f'layers.{i}.self_attn.W_k.weight'] = k_w
            new_sd[f'layers.{i}.self_attn.W_v.weight'] = v_w

        # 2. Self-Attention: 拆分 in_proj_bias
        in_proj_b = old_sd.get(f'layers.{i}.self_attn.in_proj_bias')
        if in_proj_b is not None:
            q_b, k_b, v_b = in_proj_b.chunk(3, dim=0)
            new_sd[f'layers.{i}.self_attn.W_q.bias'] = q_b
            new_sd[f'layers.{i}.self_attn.W_k.bias'] = k_b
            new_sd[f'layers.{i}.self_attn.W_v.bias'] = v_b

        # 3. out_proj
        for p in ['weight', 'bias']:
            key = f'layers.{i}.self_attn.out_proj.{p}'
            if key in old_sd:
                new_sd[f'layers.{i}.self_attn.W_o.{p}'] = old_sd[key]

        # 4. LayerNorm (相同命名)
        for p in ['weight', 'bias']:
            for norm in ['norm1', 'norm2']:
                key = f'layers.{i}.{norm}.{p}'
                if key in old_sd:
                    new_sd[f'layers.{i}.{norm}.{p}'] = old_sd[key]

        # 5. FFN: linear1 → ffn.0, linear2 → ffn.3
        ff_map = {'linear1': 'ffn.0', 'linear2': 'ffn.3'}
        for old_name, new_name in ff_map.items():
            for p in ['weight', 'bias']:
                key = f'layers.{i}.{old_name}.{p}'
                if key in old_sd:
                    new_sd[f'layers.{i}.{new_name}.{p}'] = old_sd[key]

    # 加载
    missing, unexpected = new_model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  ⚠️ 缺失参数: {[m for m in missing if 'pos_encoding' not in m]}")
    if unexpected:
        print(f"  ⚠️ 多余参数: {unexpected}")
    print(f"  ✅ 权重转换完成！")


# ============================================================
#  #4 SSD 经验回放缓冲区
# ============================================================

class ResponseBuffer:
    """
    将模型生成的回答存到 SSD，多个 epoch 复用。
    减少重复生成，提升训练效率。
    """
    def __init__(self, path: str = "/root/airesearch/v3/buffer.pt",
                 max_size: int = 5000):   # buffer 只做冷启动锚点，不用回放
        self.path = path
        self.max_size = max_size
        self.buffer = []          # [(prompt, resp_tokens, reward, gt), ...]
        self.dirty = False
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                data = torch.load(self.path)
                if isinstance(data, list):
                    self.buffer = data
                    print(f"  [Buffer] 从 SSD 加载 {len(self.buffer)} 条经验")
            except Exception as e:
                print(f"  [Buffer] 加载失败: {e}")

    def _save(self):
        if self.dirty:
            torch.save(self.buffer[-self.max_size:], self.path)
            self.dirty = False
            print(f"  [Buffer] 已保存 {min(len(self.buffer), self.max_size)} 条到 SSD")

    def add(self, prompt: str, resp_tokens: List[int], reward: float, gt: int):
        self.buffer.append((prompt, resp_tokens, reward, gt))
        # 只保留 top max_size 条（按 reward 排序淘汰）
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda x: x[2], reverse=True)
            self.buffer = self.buffer[:self.max_size]
        self.dirty = True

    def add_batch(self, prompts, responses_list, rewards, gts):
        """批量添加"""
        for p, r_tokens, r, gt in zip(prompts, responses_list, rewards, gts):
            self.buffer.append((p, r_tokens, r, gt))
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda x: x[2], reverse=True)
            self.buffer = self.buffer[:self.max_size]
        self.dirty = True

    def sample(self, n: int) -> List[Tuple]:
        """从缓冲区采样 n 条（加权采样，奖励越高概率越大）"""
        if n <= 0 or len(self.buffer) == 0:
            return []
        if len(self.buffer) <= n:
            return random.sample(self.buffer, len(self.buffer))

        # 按 reward 加权采样
        rewards = torch.tensor([item[2] for item in self.buffer])
        # 偏移确保正权重
        weights = (rewards - rewards.min() + 0.1).float()
        weights = weights / weights.sum()

        idx = torch.multinomial(weights, n, replacement=False)
        return [self.buffer[i] for i in idx]

    def save(self):
        """显式保存"""
        self._save()

    @property
    def size(self):
        return len(self.buffer)


# ============================================================
#  #1 连续链数据集
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


class ChainDataset(Dataset):
    """
    连续链 CoT 数据集，预加载到 RAM。
    每步长度 2~8 随机，操作数范围 -50~200。
    """
    def __init__(self, num_samples: int = 12000, 
                 min_steps: int = 2, max_steps: int = 8,
                 max_val: int = 200, seed: int = 123):
        self.tokenizer = CharTokenizer()
        rng = np.random.RandomState(seed)

        self.prompts: List[str] = []
        self.ground_truths: List[int] = []
        self.prompt_tokens: List[List[int]] = []
        self.cots: List[str] = []  # 完整 CoT，用于冷启动

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
            prompt = lines[0]  # 如 "12-3+5="
            gt_line = lines[-1]  # 如 "=14"
            gt = int(gt_line[1:])

            self.prompts.append(prompt)
            self.ground_truths.append(gt)
            self.prompt_tokens.append(
                self.tokenizer.encode(prompt, add_special=True))
            self.cots.append(cot)

        # 打乱
        combined = list(zip(self.prompts, self.ground_truths, 
                           self.prompt_tokens, self.cots))
        rng.shuffle(combined)
        self.prompts = [c[0] for c in combined]
        self.ground_truths = [c[1] for c in combined]
        self.prompt_tokens = [c[2] for c in combined]
        self.cots = [c[3] for c in combined]

        print(f"  [ChainDataset] {len(self.prompts)} 条连续链, "
              f"{min_steps}~{max_steps} 步, 已预加载 ✅")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.ground_truths[idx], self.prompt_tokens[idx]


# ============================================================
#  #1 采样参数调优后的生成函数
# ============================================================

@torch.no_grad()
def generate_optimized(model, prompt: str, tokenizer: CharTokenizer,
                       max_gen_len: int = 60, temperature: float = 0.8,
                       top_p: float = 0.9) -> Tuple[str, List[int]]:
    """
    优化版生成：
    - max_gen_len=60（正常 CoT 只需要 30~40 token）
    - temperature=0.8（降低随机性，更容易采到 EOS）
    - top_p=0.9（截断低概率 token）
    - 使用 KV cache
    """
    return model.generate_with_cache(prompt, tokenizer, max_gen_len, temperature, top_p)


# ============================================================
#  #3 步骤级奖励：三级 + 双目校验
# ============================================================

def compute_step_reward(response: str, ground_truth: int, prompt: str = "") -> float:
    """
    连续链 CoT 的逐步骤校验奖励（#3）。

    三级奖励（对每行 CoT A⊕B=C）：
      ✅ A 对上一步, B 对题目数, C 算对 → +0.5
      🟡 A 对, B 对, 但 C 算错         → +0.1（抄对题但算错）
      🔴 A 错（链断了）                 → -0.3
      🟠 A 对, 但 B 错（抄错题）        → -0.2

    bonus: 最终答案正确 → +0.2
    无任何 CoT 步骤 → 纯乱码 -0.5

    双目校验：
      - 左目：A == 上一步的 C（链跟踪）
      - 右目：B == 题目中按顺序的下一个操作数（原题注意力）
    """
    lines = response.strip().split('\n')
    reward = 0.0
    step_re = re.compile(r'^(-?\d+)([+-])(-?\d+)=(-?\d+)$')

    # 从 prompt 解析操作数序列和运算符序列
    # prompt 如 "12-3+5=" → nums=[12,3,5], ops=['-','+']
    prompt_nums = []
    prompt_ops = []
    if prompt:
        # 去掉末尾的 =
        p = prompt.rstrip('=')
        # 用正则提取：数字和运算符交替
        parts = re.findall(r'[+-]?\d+', p)
        if parts:
            prompt_nums.append(int(parts[0]))
            for part in parts[1:]:
                if part[0] in '+-':
                    prompt_ops.append(part[0])
                    prompt_nums.append(int(part[1:]))

    prev_c = None
    n_steps = 0
    step_idx = 0  # 当前是第几步（0-indexed）

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
                # 第一步，检查 A 是否等于题目第一个数
                if prompt_nums:
                    a_correct = a_correct and (a == prompt_nums[0])

            # 校验 C 是否算对
            if op == '+':
                c_correct = (a + b == c)
            else:
                c_correct = (a - b == c)

            # 三级 + 四色奖励
            if a_correct and b_correct and c_correct:
                reward += 0.5     # 🟢 全对
            elif a_correct and b_correct and not c_correct:
                reward += 0.1     # 🟡 跟对链但算错
            elif a_correct and not b_correct:
                reward -= 0.2     # 🟠 A 对了但 B 抄错了题目的数
            else:
                reward -= 0.3     # 🔴 A 错了，链断了

            prev_c = c
            step_idx += 1

        elif line.startswith('='):
            try:
                final = int(line[1:].strip())
                if final == ground_truth:
                    reward += 0.2  # bonus
            except ValueError:
                pass

    if n_steps == 0:
        reward -= 0.5  # 纯乱码

    return reward


# ============================================================
#  可微 Log Prob 计算（适配新模型）
# ============================================================

def compute_log_probs(model, prompt_tokens: List[int], response_tokens: List[int],
                      device: str = 'cpu') -> torch.Tensor:
    # 不强制 train()，保持模型当前模式（ref 模型 eval 时不应用 dropout）
    full_ids = prompt_tokens + response_tokens
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_tokens)

    logits = model(full_tensor)
    resp_logits = logits[0, prompt_len - 1: -1, :]
    # 裁剪 logits 防止 softmax 溢出 → NaN
    resp_logits = torch.clamp(resp_logits, min=-50, max=50)
    log_prob_dist = F.log_softmax(resp_logits, dim=-1)
    # 检查 NaN
    if torch.isnan(log_prob_dist).any():
        log_prob_dist = torch.nan_to_num(log_prob_dist, nan=-100.0)

    resp_tensor = torch.tensor(response_tokens, dtype=torch.long, device=resp_logits.device)
    gathered = log_prob_dist[torch.arange(len(response_tokens), device=resp_logits.device), resp_tensor]
    gathered = torch.clamp(gathered, min=-100, max=0)  # 防止 -inf × 0 = NaN

    return gathered


# ============================================================
#  RL 训练（单一 epoch，从缓冲区采样 + SSD 回放）
# ============================================================

def train_rl_epoch(model, tokenizer, dataset, optimizer, device,
                   response_buffer: ResponseBuffer,
                   num_samples: int = 4, temperature: float = 0.8,
                   top_p: float = 0.9, max_gen_len: int = 60,
                   max_prompts: int = 500,
                   replay_ratio: float = 0.3,
                   ref_model=None, kl_beta: float = 0.0) -> Tuple[float, float, float, float]:
    """
    RL 训练一个 epoch。
    replay_ratio: 从 SSD 回放缓冲区采样的比例（剩余从数据集生成）
    ref_model: 参考模型（SFT），用于 GRPO 无偏 KL 惩罚 ← #10
    kl_beta: KL 惩罚系数
    """
    model.eval()  # GRPO 不需要 dropout，关闭以防与 ref model 模式不一致
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:max_prompts]

    total_rewards = []
    total_policy_loss = 0.0
    total_kl = 0.0          # ← #9
    correct_count = 0
    total_count = 0

    for batch_idx, idx in enumerate(indices):
        prompt, gt, prompt_tokens = dataset[idx]

        # ---- 生成新回答 vs 从缓冲区回放 ----
        responses = []
        all_resp_tokens = []
        response_prompts = []  # 每个回答对应的 prompt（回放样本用自己原始的）
        response_gts = []      # 每个回答对应的 gt

        # 决定采几个新样本、几个回放
        n_generate = max(1, int(num_samples * (1 - replay_ratio)))
        n_replay = num_samples - n_generate

        # 生成新样本
        for _ in range(n_generate):
            resp_text, resp_tokens = generate_optimized(
                model, prompt, tokenizer,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )
            responses.append(resp_text)
            all_resp_tokens.append(resp_tokens)
            response_prompts.append(prompt)
            response_gts.append(gt)

        # 从缓冲区回放（冷启动的正确 CoT，用回放样本自己的 gt 和 prompt 算奖励）
        replay_samples = response_buffer.sample(n_replay)
        for p_buf, r_tokens, r, gt_buf in replay_samples:
            responses.append(tokenizer.decode(r_tokens))
            all_resp_tokens.append(r_tokens)
            response_prompts.append(p_buf)
            response_gts.append(gt_buf)

        # 如果回放样本不够，补生成
        while len(responses) < num_samples:
            resp_text, resp_tokens = generate_optimized(
                model, prompt, tokenizer,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )
            responses.append(resp_text)
            all_resp_tokens.append(resp_tokens)

        # ---- 计算奖励（#3 步骤级，每个回答用对应的 prompt 和 gt）----
        rewards = [compute_step_reward(resp, response_gts[i], response_prompts[i])
                   for i, resp in enumerate(responses)]

        for i, resp in enumerate(responses):
            m = re.search(r'=(-?\d+)', resp)
            if m and int(m.group(1)) == response_gts[i]:
                correct_count += 1
            total_count += 1

        # 存入 SSD 回放缓冲区（新样本和回放样本都存，但回放已存过，重复存不影响）
        response_buffer.add_batch(
            response_prompts, all_resp_tokens, rewards, response_gts
        )

        # ---- 组内 advantage（所有样本统一归一化）----
        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        # ---- 策略梯度 + KL 惩罚 ----  ← #9 (fixed sign)
        policy_loss = torch.tensor(0.0, device=device)
        kl_loss_sum = torch.tensor(0.0, device=device)
        n_kl_valid = 0
        for i in range(len(all_resp_tokens)):
            if len(all_resp_tokens[i]) == 0:
                continue
            log_probs = compute_log_probs(model, prompt_tokens, all_resp_tokens[i], device)
            policy_loss = policy_loss - advantages[i] * log_probs.sum()

            # GRPO 无偏 KL 估计：KL = exp(d) - d - 1
            if ref_model is not None and kl_beta > 0:
                with torch.no_grad():
                    ref_log_probs = compute_log_probs(ref_model, prompt_tokens, all_resp_tokens[i], device)
                d = ref_log_probs - log_probs  # 逐 token
                # 数值稳定 GRPO KL：对大 d 用近似 exp(d)≈exp(clamp(d))
                d_clamped = torch.clamp(d, min=-20, max=20)
                kl_per_token = torch.exp(d_clamped) - d - 1
                if torch.isnan(kl_per_token).any():
                    kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0)
                kl_div = kl_per_token.mean()
                kl_loss_sum = kl_loss_sum + kl_div
                n_kl_valid += 1

        policy_loss = policy_loss / num_samples
        if ref_model is not None and kl_beta > 0 and n_kl_valid > 0:
            kl_mean = kl_loss_sum / n_kl_valid
            kl_penalty = kl_beta * kl_mean  # 始终为正
            policy_loss = policy_loss + kl_penalty

        optimizer.zero_grad()
        if torch.isnan(policy_loss).any():
            policy_loss = torch.tensor(0.0, device=device)  # 跳过 NaN
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        if ref_model is not None and kl_beta > 0 and n_kl_valid > 0:
            total_kl += (kl_loss_sum / n_kl_valid).item()
        total_rewards.append(mean_r.item())

        if (batch_idx + 1) % 100 == 0:
            avg_r = np.mean(total_rewards[-100:]) if total_rewards else 0
            ts = time.strftime("%H:%M:%S")
            print(f"    [{ts}] [{batch_idx+1}/{len(indices)}]  "
                  f"Reward: {avg_r:.3f}  Loss: {policy_loss.item():.4f}  "
                  f"Buffer: {response_buffer.size}")

    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    avg_loss = total_policy_loss / max(len(indices), 1)
    accuracy = correct_count / max(total_count, 1) * 100
    avg_kl = total_kl / max(n_kl_valid, 1) if ref_model is not None and kl_beta > 0 else 0.0

    return avg_reward, avg_loss, accuracy, avg_kl


# ============================================================
#  Checkpoint 管理器（兼容原版）
# ============================================================

class CheckpointManager:
    def __init__(self, save_dir: str, model_name: str = "transformer_cot_opt"):
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
#  主训练
# ============================================================

def main():
    # ======== 配置（#1 采样参数已调优）=======
    D_MODEL     = 128
    N_LAYERS    = 4
    N_HEADS     = 4
    D_FF        = 512
    MAX_LEN     = 256
    DROPOUT     = 0.1

    BATCH_SIZE  = 64
    RL_LR       = 1e-4       # RL 学习率
    NUM_SAMPLES = 12000
    MAX_VAL     = 200        # 操作数范围

    RL_EPOCHS   = 30         # GRPO 训练轮数

    # === GRPO 超参 ===
    RL_KL_BETA      = 0.01   # GRPO 无偏 KL 惩罚系数
    RL_TEMPERATURE  = 0.7    # 采样温度
    RL_TOP_P        = 0.9    # nucleus 采样
    RL_MAX_GEN_LEN  = 100    # 连续链 CoT 需要更多 token（8 步约 50+）
    RL_NUM_SAMPLES  = 8      # GRPO 每组采样数
    RL_PROMPTS_PER_EPOCH = 500
    REPLAY_RATIO    = 0.3    # 30% 从 buffer 回放（冷启动正确 CoT 参与训练）

    # === SFT 热身 ===
    SFT_MIN_EPOCHS = 30      # 最少训练轮数（保证充分学习格式）
    SFT_MAX_EPOCHS = 200     # 最多轮数（防止无限跑）
    SFT_LR = 1e-3
    SFT_BATCH_SIZE = 32

    # === 冷启动 ===
    COLD_START_SIZE = 500    # 预填充的正确 CoT 样本数

    CKPT_DIR = "/root/airesearch/v3/checkpoints"
    LOG_PATH = "/root/airesearch/v3/train.log"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")
    print(f"存档: {CKPT_DIR}")
    print(f"日志: {LOG_PATH}")

    # ======== 模型（#2 KV cache）=======
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RL_EPOCHS
    )

    ckpt_manager = CheckpointManager(CKPT_DIR)
    start_epoch = ckpt_manager.load_latest(model, optimizer)
    if start_epoch is not None:
        start_epoch += 1
        print(f"  从 Epoch {start_epoch + 1} 继续")
    else:
        # 不加载旧 SFT 权重。用冷启动数据做 SFT 热身（下方）
        print("  ↪ 随机初始化，等待冷启动 SFT 热身")
        start_epoch = 0

    # ======== SSD 经验回放缓冲区（#4）=======
    response_buffer = ResponseBuffer()

    # ======== #1 冷启动：用规则正确 CoT 填充缓冲区 ========
    sft_data = []  # 用于 SFT 预训练
    if response_buffer.size < COLD_START_SIZE:
        print(f"\n=== 冷启动：填充 {COLD_START_SIZE} 条正确 CoT 样本… ===")
        cold_count = 0
        while response_buffer.size < COLD_START_SIZE:
            n_steps = random.randint(2, 6)
            start = random.randint(-20, 100)
            ops = [random.choice(['+', '-']) for _ in range(n_steps)]
            nums = [random.randint(0, 50) for _ in range(n_steps)]
            cot = generate_chain_cot(start, ops, nums)
            lines = cot.split('\n')
            prompt = lines[0]
            gt = int(lines[-1][1:])
            full_text = prompt + "\n".join(lines[1:])  # prompt + CoT + =result
            tokens = tokenizer.encode(cot)
            reward = compute_step_reward(cot, gt, prompt)
            response_buffer.add(prompt, tokens, reward, gt)
            sft_data.append(tokenizer.encode(cot, add_special=True))  # 含 BOS/EOS
            cold_count += 1
        print(f"  ✅ 已填充 {cold_count} 条正确 CoT")
        response_buffer.save()

    # ======== SFT 热身：用冷启动数据预训练模型（训到收敛）========
    if len(sft_data) > 0:
        print(f"\n=== SFT 热身：{len(sft_data)} 条样本，最少 {SFT_MIN_EPOCHS} epoch ===")
        model.train()
        sft_optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR)
        best_loss = float('inf')
        no_improve = 0
        for sft_epoch in range(SFT_MAX_EPOCHS):
            random.shuffle(sft_data)
            total_loss = 0.0
            n_batches = 0
            for batch_start in range(0, len(sft_data), SFT_BATCH_SIZE):
                batch = sft_data[batch_start:batch_start + SFT_BATCH_SIZE]
                max_len_batch = max(len(seq) for seq in batch)
                padded = torch.full((len(batch), max_len_batch), PAD_IDX,
                                    dtype=torch.long, device=DEVICE)
                for i, seq in enumerate(batch):
                    padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                logits = model(padded)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = padded[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=PAD_IDX
                )
                sft_optimizer.zero_grad()
                loss.backward()
                sft_optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            
            # 早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            
            status = "↓" if avg_loss < best_loss else "—"
            if (sft_epoch + 1) % 5 == 0 or sft_epoch == 0:
                print(f"  SFT Epoch {sft_epoch+1:>3}  Loss: {avg_loss:.4f}  {status}")
            
            # 满足最少 epoch 后检测早停
            if sft_epoch >= SFT_MIN_EPOCHS and no_improve >= 5:
                print(f"  ↪ 早停于 Epoch {sft_epoch+1}（连续 5 轮无改善）")
                break
        print(f"  ✅ SFT 热身完成（最终 loss: {best_loss:.4f}）")

    # ======== 参考模型（用于 KL 惩罚）——深拷贝 SFT 后的模型 ========
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
        print(f"  ✅ 参考模型已冻结（SFT 锚点）")

    # ======== 为 RL 阶段重建优化器（降低学习率）========  ← #9
    # 始终使用 RL_LR，无论从 checkpoint 恢复还是从头开始
    optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RL_EPOCHS
    )

    # ======== #1 连续链数据集 ========
    print("\n=== 生成连续链数据集… ===")
    rl_dataset = ChainDataset(
        num_samples=NUM_SAMPLES,
        min_steps=2, max_steps=8,
        max_val=MAX_VAL, seed=123
    )

    # ======== RL Fine-tune ========
    rl_start = max(start_epoch, 0)
    print(f"\n{'='*60}")
    print(f"🎯 GRPO + 连续链 CoT（v3）")
    print(f"   采样: T={RL_TEMPERATURE}, top_p={RL_TOP_P}, max_len={RL_MAX_GEN_LEN}")
    print(f"   回放: {REPLAY_RATIO*100:.0f}% 来自 SSD 缓冲区")
    print(f"   KV Cache: ✅ 已启用")
    print(f"   KL 惩罚: β={RL_KL_BETA} ✅ 已启用" if ref_model else f"   KL 惩罚: ❌ 未启用")
    print(f"   冷启动: {response_buffer.size} 条 ✅")
    print(f"{'='*60}")

    for rl_epoch in range(rl_start, RL_EPOCHS):
        epoch_start = time.time()

        avg_reward, avg_loss, accuracy, avg_kl = train_rl_epoch(
            model, tokenizer, rl_dataset, optimizer, DEVICE,
            response_buffer=response_buffer,
            num_samples=RL_NUM_SAMPLES,
            temperature=RL_TEMPERATURE,
            top_p=RL_TOP_P,
            max_gen_len=RL_MAX_GEN_LEN,
            max_prompts=RL_PROMPTS_PER_EPOCH,
            replay_ratio=REPLAY_RATIO,
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
            'buffer_size': response_buffer.size,
        }
        ckpt_manager.save(model, optimizer, rl_epoch, metrics, is_best=is_best)

        kl_str = f"  |  KL: {avg_kl:.4f}" if ref_model else ""
        print(f"  RL Epoch {rl_epoch+1}/{RL_EPOCHS}  |  "
              f"Reward: {avg_reward:.4f}  |  Loss: {avg_loss:.4f}  |  "
              f"Acc: {accuracy:.1f}%{kl_str}  |  {elapsed:.1f}s  |  "
              f"Buffer: {response_buffer.size}")

        # 定期评估（用 ChainDataset 采几个测试样本）
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

        # 每轮保存缓冲区
        response_buffer.save()

    # ======== 最终保存 ========
    final_path = f"{CKPT_DIR}/transformer_cot_opt_final.pth"
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

    print("\n✅ 训练完成！")


if __name__ == "__main__":
    # 重定向日志
    import sys
    from io import StringIO
    log_path = "/root/airesearch/v3/train.log"
    print(f"日志将写入: {log_path}")
    main()
