"""
v2_rl.py - R1 风格强化学习训练
================================
路线：仿 DeepSeek-R1 思想
  阶段一: SFT Warmup（让模型学会 CoT 格式和行为克隆）
  阶段二: REINFORCE RL（答案正确性信号驱动推理能力涌现）

奖励设计：
  +1.0  答案完全正确
  -0.5  答案错误
  +0.2  CoT 格式正确（包含 →↑↓ 符号）
  +0.1  CoT 结构完整（多行 CoT 步骤）

Checkpoint: 每 epoch 自动存档，支持断点续训
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
from pathlib import Path
from typing import List, Tuple, Optional

# ============================================================
#  复用 v2.py 的基础组件（Tokenizer, 模型, CoT 生成函数）
# ============================================================
# v2.py 中 main() 在 __name__=="__main__" 下，import 安全
import sys
sys.path.insert(0, '/root/airesearch/v2')

from v2 import (
    CharTokenizer, PAD_IDX, BOS_IDX, EOS_IDX,
    TransformerCoT, PositionalEncoding,
    generate_add_cot, generate_sub_cot,
    ArithmeticCoTDataset,
    evaluate_model, test_generalization
)


# ============================================================
#  1. RL 数据集：只生成 prompt + ground truth
# ============================================================

class RLDataset(Dataset):
    """
    只提供 prompt 和答案，不提供完整 CoT。
    模型自己探索推理路径，RL 信号驱动。
    """
    def __init__(self, max_val: int = 100, num_samples: int = 10000,
                 op: str = 'both', seed: int = 123):
        self.tokenizer = CharTokenizer()
        rng = np.random.RandomState(seed)

        self.prompts: List[str] = []
        self.ground_truths: List[int] = []

        # 加法
        if op in ('add', 'both'):
            n_add = num_samples // 2 if op == 'both' else num_samples
            for _ in range(n_add):
                a = rng.randint(0, max_val + 1)
                b = rng.randint(0, max_val + 1)
                self.prompts.append(f"{a}+{b}=")
                self.ground_truths.append(a + b)

        # 减法
        if op in ('sub', 'both'):
            n_sub = num_samples // 2 if op == 'both' else num_samples
            for _ in range(n_sub):
                a = rng.randint(0, max_val + 1)
                b = rng.randint(0, max_val + 1)
                self.prompts.append(f"{a}-{b}=")
                self.ground_truths.append(a - b)

        # 打乱
        combined = list(zip(self.prompts, self.ground_truths))
        rng.shuffle(combined)
        self.prompts, self.ground_truths = zip(*combined) if combined else ([], [])
        self.prompts = list(self.prompts)
        self.ground_truths = list(self.ground_truths)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.ground_truths[idx]


# ============================================================
#  2. 奖励函数
# ============================================================

def compute_reward(response: str, ground_truth: int) -> float:
    """
    计算单个回答的奖励。
    仿 R1：规则奖励 = 正确性 + 格式 + 结构完整性
    """
    reward = 0.0

    # --- 1. 答案正确性奖励（核心信号）---
    match = re.search(r'=(-?\d+)', response)
    if match:
        pred = int(match.group(1))
        reward += 1.0 if pred == ground_truth else -0.5
    else:
        reward -= 0.5  # 没输出答案，惩罚

    # --- 2. 格式奖励（引导 CoT 格式）---
    if '→' in response and '↑' in response:
        reward += 0.2

    # --- 3. CoT 结构完整性（至少 2 行 CoT 步骤）---
    cot_lines = [l for l in response.split('\n') if '→' in l and '↑' in l]
    if len(cot_lines) >= 2:
        reward += 0.1

    return reward


# ============================================================
#  3. REINFORCE 所需的生成与可微 log_prob 计算
# ============================================================
# 标准 REINFORCE trick：
#   采样（选择动作）→ 不记梯度（torch.no_grad）
#   算 log_prob（评估动作好坏）→ 重新过前向，记梯度！

@torch.no_grad()
def generate_tokens(model, prompt: str, tokenizer: CharTokenizer,
                    max_gen_len: int = 200, temperature: float = 1.0,
                    device: str = 'cpu') -> Tuple[str, List[int]]:
    """
    采样生成回答（不记梯度）。
    返回: (decoded_text, token_ids_list)
    """
    model.eval()

    input_ids = torch.tensor(
        tokenizer.encode(prompt, add_special=True),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    generated_ids = []

    for _ in range(max_gen_len):
        src_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        logits = model(input_ids, src_mask)
        next_logits = logits[0, -1, :]

        scaled = next_logits / max(temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        generated_ids.append(next_token)

        if next_token == EOS_IDX:
            break

        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

    text = tokenizer.decode(generated_ids)
    return text, generated_ids


def compute_log_probs(model, prompt_tokens: List[int], response_tokens: List[int],
                      tokenizer: CharTokenizer, device: str = 'cpu') -> torch.Tensor:
    """
    在完整序列上做一次可微前向，计算 response 部分每个 token 的 log_prob。
    返回: (seq_len,) 的 log probability 张量，有 grad_fn。
    """
    model.train()  # 启用梯度
    full_ids = prompt_tokens + response_tokens
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_tokens)

    src_mask = torch.ones_like(full_tensor, dtype=torch.bool, device=device)
    logits = model(full_tensor, src_mask)  # (1, full_len, vocab)

    # logits[t] 预测 token[t+1]
    # response 部分：logits[prompt_len-1 : -1] 对应 response[0 : ]
    resp_logits = logits[0, prompt_len - 1 : -1, :]  # (resp_len, vocab)
    log_probs = F.log_softmax(resp_logits, dim=-1)    # (resp_len, vocab)

    # 收集实际采到的 token 的 log_prob
    resp_tensor = torch.tensor(response_tokens, dtype=torch.long, device=resp_logits.device)
    gathered = log_probs[torch.arange(len(response_tokens), device=resp_logits.device), resp_tensor]

    return gathered


# ============================================================
#  4. REINFORCE RL 训练（一个 epoch）
# ============================================================

def train_rl_epoch(model, tokenizer, dataset, optimizer, device,
                   num_samples: int = 8, temperature: float = 1.0,
                   max_gen_len: int = 200,
                   max_prompts: int = 500) -> Tuple[float, float, float]:
    """
    一个 epoch 的 REINFORCE（带组内 baseline）。
    
    流程：
      对每个 prompt:
        1. 采 num_samples 个回答
        2. 算每个回答的奖励
        3. 组内 advantage = (reward - mean) / std
        4. 策略梯度：loss = -Σ(advantage_i * log_prob_i)
    
    返回: (avg_reward, avg_loss, accuracy)
    """
    model.train()
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:max_prompts]

    total_rewards = []
    total_policy_loss = 0.0
    correct_count = 0
    total_count = 0

    for batch_idx, idx in enumerate(indices):
        prompt, gt = dataset[idx]
        prompt_tokens = tokenizer.encode(prompt, add_special=True)

        # ---- 1. 采样 num_samples 个回答（不记梯度）----
        responses = []
        all_resp_tokens = []

        for _ in range(num_samples):
            resp_text, resp_tokens = generate_tokens(
                model, prompt, tokenizer,
                max_gen_len=max_gen_len,
                temperature=temperature,
                device=device
            )
            responses.append(resp_text)
            all_resp_tokens.append(resp_tokens)

        # ---- 2. 计算奖励 ----
        rewards = [compute_reward(resp, gt) for resp in responses]

        # 统计正确数
        for resp in responses:
            m = re.search(r'=(-?\d+)', resp)
            if m and int(m.group(1)) == gt:
                correct_count += 1
            total_count += 1

        # ---- 3. 组内 advantage（GRPO 风格） ----
        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        # ---- 4. 可微 log_prob + 策略梯度更新 ----
        # 重新前向计算 log_prob（记梯度）
        policy_loss = torch.tensor(0.0, device=device)
        for i in range(num_samples):
            if len(all_resp_tokens[i]) == 0:
                continue
            log_probs = compute_log_probs(
                model, prompt_tokens, all_resp_tokens[i],
                tokenizer, device
            )
            policy_loss = policy_loss - advantages[i] * log_probs.sum()

        policy_loss = policy_loss / num_samples

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_rewards.append(mean_r.item())

        # 进度打印
        if (batch_idx + 1) % 100 == 0:
            avg_r = np.mean(total_rewards[-100:]) if total_rewards else 0
            print(f"    [{batch_idx+1}/{len(indices)}]  AvgReward: {avg_r:.3f}  "
                  f"Loss: {policy_loss.item():.3f}")

    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    avg_loss = total_policy_loss / max(len(indices), 1)
    accuracy = correct_count / max(total_count, 1) * 100

    return avg_reward, avg_loss, accuracy


# ============================================================
#  5. Checkpoint 管理器
# ============================================================

class CheckpointManager:
    """管理训练存档点，支持断点续训"""
    def __init__(self, save_dir: str, model_name: str = "transformer_cot_rl"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_reward = -float('inf')
        self.best_epoch = -1

    def save(self, model, optimizer, epoch, metrics: dict, is_best: bool = False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_reward': self.best_reward,
            'best_epoch': self.best_epoch,
            'metrics': metrics,
        }

        # 最新存档（用于断点续训）
        latest_path = self.save_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        # 每 5 epoch 保留一个命名存档
        if epoch % 5 == 0:
            path = self.save_dir / f"{self.model_name}_epoch{epoch}.pth"
            torch.save(checkpoint, path)
            print(f"    [Checkpoint] 已保存: {path.name}")

        # 最佳模型
        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"    [Checkpoint] 🏆 新最佳模型! Reward: {metrics.get('avg_reward', 0):.4f}")

        # 清理旧存档
        self._cleanup(keep_last=10)

    def load_latest(self, model, optimizer=None) -> Optional[dict]:
        """加载最新 checkpoint，返回状态信息"""
        latest_path = self.save_dir / f"{self.model_name}_latest.pth"
        if not latest_path.exists():
            return None

        checkpoint = torch.load(latest_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"  [警告] 优化器状态加载失败: {e}")

        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)
        epoch = checkpoint['epoch']
        print(f"  [Checkpoint] 恢复训练于 Epoch {epoch + 1}")
        return checkpoint

    def _cleanup(self, keep_last: int = 10):
        """清理旧存档"""
        pattern = f"{self.model_name}_epoch*.pth"
        files = sorted(self.save_dir.glob(pattern))
        for f in files[:-keep_last]:
            f.unlink()


# ============================================================
#  6. 主训练流程
# ============================================================

def train():
    # ======== 配置 ========
    # 模型架构
    D_MODEL     = 128
    N_LAYERS    = 4
    N_HEADS     = 4
    D_FF        = 512
    MAX_LEN     = 256
    DROPOUT     = 0.1

    # 训练基础
    BATCH_SIZE  = 64
    BASE_LR     = 5e-4
    NUM_SAMPLES = 12000       # 训练样本数
    MAX_VAL     = 100          # 训练范围 0~100

    # SFT warmup
    SFT_EPOCHS  = 8            # 先做 8 轮行为克隆

    # RL fine-tune
    RL_EPOCHS   = 30           # RL 训练轮数
    RL_NUM_SAMPLES  = 4        # 每个 prompt 采几个回答
    RL_TEMPERATURE  = 1.0      # 采样温度
    RL_MAX_GEN_LEN  = 200      # 最大生成长度
    RL_PROMPTS_PER_EPOCH = 500 # 每轮处理多少个 prompt（子采样加速）

    # 路径
    V2_DIR = "/root/airesearch/v2"
    CKPT_DIR = f"{V2_DIR}/checkpoints"
    LOG_PATH = f"{V2_DIR}/train_rl.log"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"工作目录: {V2_DIR}")
    print(f"存档目录: {CKPT_DIR}")

    tokenizer = CharTokenizer()

    # ======== 构建模型 ========
    print("\n=== 构建模型… ===")
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

    # ======== 优化器 & 调度器 ========
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    total_epochs = SFT_EPOCHS + RL_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    # ======== Checkpoint ========
    ckpt_manager = CheckpointManager(CKPT_DIR)
    start_epoch = 0
    loaded = ckpt_manager.load_latest(model, optimizer)
    if loaded is not None:
        start_epoch = loaded['epoch'] + 1
        print(f"  从 Epoch {start_epoch + 1} 继续训练")
    else:
        print("  从头开始训练")

    # ======== 阶段一：SFT Warmup ========
    if start_epoch < SFT_EPOCHS:
        print("\n" + "=" * 60)
        print("🔥 阶段一：SFT Warmup（行为克隆 — 学会 CoT 格式）")
        print("=" * 60)

        sft_dataset = ArithmeticCoTDataset(
            max_val=MAX_VAL, num_samples=NUM_SAMPLES,
            op='both', seed=42
        )
        sft_loader = DataLoader(
            sft_dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        for epoch in range(max(start_epoch, 0), SFT_EPOCHS):
            model.train()
            total_loss = 0.0
            total_tokens = 0
            epoch_start = time.time()

            for tokens, mask in sft_loader:
                tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)

                inp = tokens[:, :-1]
                tgt = tokens[:, 1:]
                tgt_mask = mask[:, 1:]

                logits = model(inp, mask[:, :-1])

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction='none'
                )
                loss = loss.reshape_as(tgt)
                loss = loss * tgt_mask
                num_tok = tgt_mask.sum().item()
                if num_tok == 0:
                    continue
                batch_loss = loss.sum() / num_tok

                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += batch_loss.item() * num_tok
                total_tokens += num_tok

            scheduler.step()
            avg_loss = total_loss / max(total_tokens, 1)
            elapsed = time.time() - epoch_start

            metrics = {'loss': avg_loss, 'phase': 'sft', 'epoch': epoch}
            ckpt_manager.save(model, optimizer, epoch, metrics)

            print(f"  SFT Epoch {epoch+1}/{SFT_EPOCHS}  |  "
                  f"Loss: {avg_loss:.6f}  |  {elapsed:.1f}s  |  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # ======== 阶段二：RL Fine-tune ========
    if start_epoch < SFT_EPOCHS + RL_EPOCHS:
        print("\n" + "=" * 60)
        print("🎯 阶段二：REINFORCE RL Fine-tune（R1 风格）")
        print("=" * 60)

        rl_dataset = RLDataset(
            max_val=MAX_VAL, num_samples=NUM_SAMPLES,
            op='both', seed=123
        )
        print(f"  RL 数据集大小: {len(rl_dataset)}")
        print(f"  每 prompt 采样: {RL_NUM_SAMPLES} 次")
        print(f"  每 epoch 处理: {RL_PROMPTS_PER_EPOCH} prompts")

        rl_start = max(start_epoch - SFT_EPOCHS, 0)

        for rl_epoch in range(rl_start, RL_EPOCHS):
            overall_epoch = SFT_EPOCHS + rl_epoch
            epoch_start = time.time()

            avg_reward, avg_loss, accuracy = train_rl_epoch(
                model, tokenizer, rl_dataset, optimizer, DEVICE,
                num_samples=RL_NUM_SAMPLES,
                temperature=RL_TEMPERATURE,
                max_gen_len=RL_MAX_GEN_LEN,
                max_prompts=RL_PROMPTS_PER_EPOCH
            )

            scheduler.step()
            elapsed = time.time() - epoch_start

            # 是否新最佳
            is_best = avg_reward > ckpt_manager.best_reward
            if is_best:
                ckpt_manager.best_reward = avg_reward
                ckpt_manager.best_epoch = overall_epoch

            metrics = {
                'avg_reward': avg_reward,
                'rl_loss': avg_loss,
                'accuracy': accuracy,
                'phase': 'rl',
                'epoch': overall_epoch
            }
            ckpt_manager.save(model, optimizer, overall_epoch, metrics, is_best=is_best)

            print(f"  RL Epoch {rl_epoch+1}/{RL_EPOCHS}  |  "
                  f"Reward: {avg_reward:.4f}  |  Loss: {avg_loss:.4f}  |  "
                  f"Acc: {accuracy:.1f}%  |  {elapsed:.1f}s  |  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # 每 5 轮做一次全面评估
            if (rl_epoch + 1) % 5 == 0:
                print("\n  " + "-" * 50)
                print("  📊 中间评估:")
                evaluate_model(model, tokenizer, device=DEVICE,
                               num_tests=15, max_val=MAX_VAL)
                test_generalization(model, tokenizer, device=DEVICE)
                print("  " + "-" * 50 + "\n")

    # ======== 最终保存 ========
    final_path = f"{CKPT_DIR}/transformer_cot_rl_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 最终模型权重: {final_path}")

    # ======== 最终评估 ========
    print("\n" + "=" * 60)
    print("🏁 最终评估")
    print("=" * 60)
    evaluate_model(model, tokenizer, device=DEVICE,
                   num_tests=20, max_val=MAX_VAL)
    test_generalization(model, tokenizer, device=DEVICE)

    print("\n✅ 训练完成！")


if __name__ == "__main__":
    train()
