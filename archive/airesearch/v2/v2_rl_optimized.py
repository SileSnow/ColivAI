"""
v2_rl_optimized.py — 综合优化版
================================
整合优化项 #1~#5（逐条标注）

#1 ✅ 采样参数调优: temperature=0.8, top_p=0.9, max_gen_len=60
#2 ✅ KV Cache 加速: CausalSelfAttentionWithCache
#3 ✅ 修复 CoT 数据: 用 v2.py 中已修好的 generate_add_cot
#4 ✅ SSD 经验回放: ResponseBuffer 存到 /root/airesearch/v2/buffer.pt
#5 ✅ RAM 预加载: PreloadedRLDataset 一次加载全部数据到内存
#10 ✅ GRPO 训练: 组内 advantage + 无偏 KL 估计器

依赖: 可加载 v2.py SFT 训练的 checkpoint 转换后使用
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
    PositionalEncoding,
    evaluate_model, test_generalization
)

# ============================================================
#  引用已修复的 CoT 数据生成函数  ← #3
# ============================================================
# v2.py 中的 generate_add_cot / generate_sub_cot 已修复 carry/borrow 分离
# 而 datagen.py 中的版本有 bug（变量名混淆）
from v2 import generate_add_cot, generate_sub_cot


# ============================================================
#  #2 带 KV Cache 的自注意力 & Decoder 层
# ============================================================

class CausalSelfAttentionWithCache(nn.Module):
    """
    因果自注意力，支持 KV cache。
    权重命名兼容 nn.MultiheadAttention，可用于加载 SFT 权重。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # 注意：这里用 bias=True 以兼容 nn.MultiheadAttention 的权重结构
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

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

        total_len = K.size(2)
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
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayerWithCache(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_mask=None, past_kv=None, return_kv=False):
        """
        x: (batch, seq_len) token ids
        past_kv: list of [(K, V), ...] for each layer
        return_kv: 是否返回 KV cache
        """
        seq_len = x.size(1)
        h = self.token_embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)
        h = self.dropout(h)

        new_kv = []
        for i, layer in enumerate(self.layers):
            k, v = past_kv[i] if past_kv is not None else (None, None)
            h, new_k, new_v = layer(h, k, v)
            new_kv.append((new_k, new_v))

        h = self.norm(h)
        logits = self.output_proj(h)

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

            # top_p 采样
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
            probs = filtered / filtered.sum()

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
    def __init__(self, path: str = "/root/airesearch/v2/buffer.pt",
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
#  #5 RAM 预加载数据集
# ============================================================

class PreloadedRLDataset(Dataset):
    """
    RL 数据集：生成所有 prompt 和 gt 后预加载到 RAM。
    训练时零开销（直接从内存取）。
    """
    def __init__(self, max_val: int = 100, num_samples: int = 10000,
                 op: str = 'both', seed: int = 123):
        self.tokenizer = CharTokenizer()
        rng = np.random.RandomState(seed)

        self.prompts: List[str] = []
        self.ground_truths: List[int] = []
        # 预 tokenize 到 RAM
        self.prompt_tokens: List[List[int]] = []

        # 加法
        if op in ('add', 'both'):
            n_add = num_samples // 2 if op == 'both' else num_samples
            for _ in range(n_add):
                a = int(rng.randint(0, max_val + 1))
                b = int(rng.randint(0, max_val + 1))
                prompt = f"{a}+{b}="
                self.prompts.append(prompt)
                self.ground_truths.append(a + b)
                self.prompt_tokens.append(self.tokenizer.encode(prompt, add_special=True))

        # 减法
        if op in ('sub', 'both'):
            n_sub = num_samples // 2 if op == 'both' else num_samples
            for _ in range(n_sub):
                a = int(rng.randint(0, max_val + 1))
                b = int(rng.randint(0, max_val + 1))
                prompt = f"{a}-{b}="
                self.prompts.append(prompt)
                self.ground_truths.append(a - b)
                self.prompt_tokens.append(self.tokenizer.encode(prompt, add_special=True))

        # 打乱
        combined = list(zip(self.prompts, self.ground_truths, self.prompt_tokens))
        rng.shuffle(combined)
        self.prompts = [c[0] for c in combined]
        self.ground_truths = [c[1] for c in combined]
        self.prompt_tokens = [c[2] for c in combined]

        print(f"  [RAM] 数据集: {len(self.prompts)} 条, 预加载到内存 ✅")

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
#  奖励函数（同原版）
# ============================================================

def compute_reward(response: str, ground_truth: int) -> float:
    reward = 0.0
    match = re.search(r'=(-?\d+)', response)
    if match:
        pred = int(match.group(1))
        reward += 1.0 if pred == ground_truth else -0.5
    else:
        # 至少有 = 号但没抽到数字 → 部分奖励（鼓励模型输出 =）
        if '=' in response:
            reward -= 0.3
        else:
            reward -= 0.5  # 完全没有 = → 最差

    if '→' in response and '↑' in response:
        reward += 0.2

    cot_lines = [l for l in response.split('\n') if '→' in l and '↑' in l]
    if len(cot_lines) >= 2:
        reward += 0.1

    return reward


# ============================================================
#  可微 Log Prob 计算（适配新模型）
# ============================================================

def compute_log_probs(model, prompt_tokens: List[int], response_tokens: List[int],
                      device: str = 'cpu') -> torch.Tensor:
    model.train()
    full_ids = prompt_tokens + response_tokens
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_tokens)

    logits = model(full_tensor)
    resp_logits = logits[0, prompt_len - 1: -1, :]
    log_prob_dist = F.log_softmax(resp_logits, dim=-1)

    resp_tensor = torch.tensor(response_tokens, dtype=torch.long, device=resp_logits.device)
    gathered = log_prob_dist[torch.arange(len(response_tokens), device=resp_logits.device), resp_tensor]

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
    model.train()
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
        new_samples = []

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

        # 从缓冲区回放
        replay_samples = response_buffer.sample(n_replay)
        for p, r_tokens, r, gt_val in replay_samples:
            # 确保回放样本的 gt 匹配（近似匹配当前 prompt）
            # 如果 gt 不同，跳过
            if gt_val == gt:
                responses.append(tokenizer.decode(r_tokens))
                all_resp_tokens.append(r_tokens)

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

        # ---- 计算奖励 ----
        rewards = [compute_reward(resp, gt) for resp in responses]

        for resp in responses:
            m = re.search(r'=(-?\d+)', resp)
            if m and int(m.group(1)) == gt:
                correct_count += 1
            total_count += 1

        # 存入 SSD 回放缓冲区
        response_buffer.add_batch(
            [prompt] * len(all_resp_tokens),
            all_resp_tokens, rewards, [gt] * len(all_resp_tokens)
        )

        # ---- 组内 advantage ----
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

            # GRPO 无偏 KL 估计：KL = exp(d) - d - 1, 其中 d = ref_log_prob - log_prob
            # 始终 ≥ 0（不需要 abs hack），且对双向偏离都惩罚
            if ref_model is not None and kl_beta > 0:
                with torch.no_grad():
                    ref_log_probs = compute_log_probs(ref_model, prompt_tokens, all_resp_tokens[i], device)
                d = ref_log_probs - log_probs  # 逐 token
                kl_per_token = torch.exp(d) - d - 1  # 始终 ≥ 0
                kl_div = kl_per_token.mean()
                kl_loss_sum = kl_loss_sum + kl_div
                n_kl_valid += 1

        policy_loss = policy_loss / num_samples
        if ref_model is not None and kl_beta > 0 and n_kl_valid > 0:
            kl_mean = kl_loss_sum / n_kl_valid
            kl_penalty = kl_beta * kl_mean  # 始终为正
            policy_loss = policy_loss + kl_penalty

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        if ref_model is not None and kl_beta > 0 and n_kl_valid > 0:
            total_kl += (kl_loss_sum / n_kl_valid).item()
        total_rewards.append(mean_r.item())

        if (batch_idx + 1) % 100 == 0:
            avg_r = np.mean(total_rewards[-100:]) if total_rewards else 0
            print(f"    [{batch_idx+1}/{len(indices)}]  "
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
    BASE_LR     = 5e-4       # SFT 学习率（未使用，仅兼容）
    RL_LR       = 1e-4       # RL 学习率 ← #9 降低防漂移
    NUM_SAMPLES = 12000
    MAX_VAL     = 100

    SFT_EPOCHS  = 8          # SFT warmup
    RL_EPOCHS   = 30         # RL fine-tune

    # === RL 超参 ===
    RL_KL_BETA      = 0.01   # ← #10 GRPO KL 惩罚系数（无偏估计，更小即可）
    RL_TEMPERATURE  = 0.7    # 适中温度，平衡探索与利用
    RL_TOP_P        = 0.9    # nucleus 采样，截断低概率 token
    RL_MAX_GEN_LEN  = 60     # 正常 CoT 只需 ~30 token，给余量
    RL_NUM_SAMPLES  = 8      # ← #10 GRPO 每组采样数
    RL_PROMPTS_PER_EPOCH = 500
    REPLAY_RATIO    = 0.0    # ← 全 on-policy，buffer 回放关掉（旧样本过时了）

    # === 冷启动参数 ===  ← #8
    COLD_START_SIZE = 500    # 预填充的正确 CoT 样本数

    V2_DIR = "/root/airesearch/v2"
    CKPT_DIR = f"{V2_DIR}/checkpoints_opt"
    LOG_PATH = f"{V2_DIR}/train_opt.log"

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SFT_EPOCHS + RL_EPOCHS
    )

    ckpt_manager = CheckpointManager(CKPT_DIR)
    start_epoch = ckpt_manager.load_latest(model, optimizer)
    if start_epoch is not None:
        start_epoch += 1
        print(f"  从 Epoch {start_epoch + 1} 继续")
    else:
        # 尝试从原版 SFT checkpoint 转换
        sft_path = f"{V2_DIR}/checkpoints/transformer_cot_rl_epoch5.pth"
        if os.path.exists(sft_path):
            print("\n=== 转换原版 SFT 权重… ===")
            convert_sft_checkpoint(sft_path, model, DEVICE)
            start_epoch = 0
            print("  ✅ SFT 权重加载完成，开始 RL 训练")
        else:
            print("  ⚠️ 未找到 SFT checkpoint，从头随机初始化")
            start_epoch = 0

    # ======== SSD 经验回放缓冲区（#4）=======
    response_buffer = ResponseBuffer()

    # ======== #8 冷启动：用规则正确 CoT 填充缓冲区 ========
    if response_buffer.size < COLD_START_SIZE:
        print(f"\n=== #8 冷启动：填充 {COLD_START_SIZE} 条正确 CoT 样本… ===")
        cold_count = 0
        while response_buffer.size < COLD_START_SIZE:
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
            if random.random() < 0.5:
                cot = generate_add_cot(a, b)
                gt = a + b
                prompt = f"{a}+{b}="
            else:
                cot = generate_sub_cot(a, b)
                gt = a - b
                prompt = f"{a}-{b}="
            tokens = tokenizer.encode(cot)
            reward = compute_reward(cot, gt)
            response_buffer.add(prompt, tokens, reward, gt)
            cold_count += 1
        print(f"  ✅ 已填充 {cold_count} 条正确 CoT 到缓冲区 (reward=+1.3)")
        response_buffer.save()

    # ======== #9 加载参考模型（SFT）用于 KL 惩罚 ========
    ref_model = None
    if RL_KL_BETA > 0:
        print("\n=== #9 加载参考模型（SFT）用于 KL 惩罚… ===")
        ref_model = TransformerCoTWithCache(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL, n_layers=N_LAYERS,
            n_heads=N_HEADS, d_ff=D_FF,
            max_len=MAX_LEN, dropout=DROPOUT
        )
        # 从 SFT checkpoint 加载（和主模型同样的权重）
        sft_path = f"{V2_DIR}/checkpoints/transformer_cot_rl_epoch5.pth"
        convert_sft_checkpoint(sft_path, ref_model, DEVICE)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"  ✅ 参考模型已冻结")

    # ======== 为 RL 阶段重建优化器（降低学习率）========  ← #9
    # 始终使用 RL_LR，无论从 checkpoint 恢复还是从头开始
    optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RL_EPOCHS
    )

    # ======== RAM 预加载数据集（#5）=======
    print("\n=== 加载 RL 数据集到 RAM… ===")
    rl_dataset = PreloadedRLDataset(
        max_val=MAX_VAL, num_samples=NUM_SAMPLES, op='both', seed=123
    )

    # ======== RL Fine-tune ========
    rl_start = max(start_epoch, 0)
    print(f"\n{'='*60}")
    print(f"🎯 GRPO Fine-tune（#10）")
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

        # 定期评估
        if (rl_epoch + 1) % 5 == 0:
            print("\n  " + "-" * 50)
            print("  📊 中间评估:")
            evaluate_model(model, tokenizer, device=DEVICE,
                          num_tests=15, max_val=MAX_VAL)
            print("  " + "-" * 50 + "\n")

        # 每轮保存缓冲区
        response_buffer.save()

    # ======== 最终保存 ========
    final_path = f"{CKPT_DIR}/transformer_cot_opt_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 最终模型: {final_path}")

    print("\n" + "=" * 60)
    print("🏁 最终评估")
    print("=" * 60)
    evaluate_model(model, tokenizer, device=DEVICE, num_tests=20, max_val=MAX_VAL)

    print("\n✅ 训练完成！")


if __name__ == "__main__":
    # 重定向日志
    import sys
    from io import StringIO
    log_path = "/root/airesearch/v2/train_opt.log"
    print(f"日志将写入: {log_path}")
    main()
