"""
GerminalSpark — 主架构

组装所有模块：
├── GerminalPool      — 干细胞池
├── ColumnPool        — 皮质柱池
├── CorpusGate        — 门控路由
├── HippocampalHub    — 海马体
└── SparkTrigger      — 生长触发器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from config import SparkConfig
from germinal_pool import GerminalPool
from spark_column import SparkColumn, Neuroblast
from corpus_gate import CorpusGate
from hippocampal_hub import HippocampalHub
from spark_trigger import SparkTrigger


class ColumnPool(nn.Module):
    """
    皮质柱池——管理所有活跃柱

    包含：
    - 成熟的 SparkColumn（4 层）
    - 正在生长的 Neuroblast（0-3 层）
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int,
                 max_columns: int, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.max_columns = max_columns

        # 活跃的柱列表
        self.columns: List[SparkColumn] = []
        self.neuroblasts: List[Neuroblast] = []

    def add_spark_column(self, column: SparkColumn):
        """添加一个成熟柱"""
        self.columns.append(column)

    def add_neuroblast(self, nb: Neuroblast):
        """添加一个未成熟柱"""
        self.neuroblasts.append(nb)

    def grow_all(self) -> List[str]:
        """
        对所有 Neuroblast 尝试 grow() 一步
        成熟的迁移到 columns 列表
        Returns: 生长日志列表
        """
        logs = []
        new_neuroblasts = []

        for nb in self.neuroblasts:
            still_growing = nb.grow()
            log = f"  🌱 Neuroblast[{id(nb)%1000}]: layers={nb.num_layers}"
            if nb.is_mature:
                self.columns.append(nb)
                log += " → 成熟！迁移至 ColumnPool"
            else:
                new_neuroblasts.append(nb)
            logs.append(log)

        self.neuroblasts = new_neuroblasts
        return logs

    @property
    def all_columns(self) -> List[SparkColumn]:
        """所有柱（成熟 + 未成熟）"""
        return self.columns + self.neuroblasts

    @property
    def active_count(self) -> int:
        return len(self.all_columns)

    @property
    def mature_count(self) -> int:
        return len(self.columns)

    def __len__(self):
        return self.active_count


class GerminalSpark(nn.Module):
    """
    Germinal Spark — 自适应生长式 MoE 架构

    训练流程:
    1. forward() → 前向传播，海马体缓存
    2. consolidate() → 海马体蒸馏（定期）
    3. check_growth() → 检查触发 → spark()/grow()
    """

    def __init__(self, config: SparkConfig):
        super().__init__()
        self.cfg = config

        # === Token Embedding（母 embedding）===
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # === GerminalPool（干细胞池）===
        self.germinal_pool = GerminalPool(
            config.d_model, config.d_ff, config.n_heads,
            num_cells=config.germ_cells, max_len=config.max_seq_len
        )

        # === ColumnPool（皮质柱池）===
        self.column_pool = ColumnPool(
            config.d_model, config.d_ff, config.n_heads,
            max_columns=config.max_columns, max_len=config.max_seq_len
        )

        # === 初始化初始皮质柱 ===
        for _ in range(config.initial_columns):
            nb = self.germinal_pool.spark(noise=0.1)
            self.column_pool.add_neuroblast(nb)

        # 初次生长（让初始柱至少有一些层）
        for _ in range(3):  # 先长 3 步，让初始柱有全部4层（含L23注意力）
            self.column_pool.grow_all()

        # === CorpusGate（门控）===
        self.corpus_gate = CorpusGate(
            config.d_model, config.max_columns,
            top_k=config.top_k, noise_std=config.noise_std,
            load_balance_alpha=config.load_balance_alpha
        )

        # === HippocampalHub（海马体）===
        self.hippocampus = HippocampalHub(
            config.d_model,
            buffer_size=config.episodic_buffer_size,
            distill_lr=config.hippo_lr
        )

        # === SparkTrigger（生长触发器）===
        self.trigger = SparkTrigger(
            pe_dormancy_threshold=config.pe_dormancy_threshold,
            pe_patience=config.pe_dormancy_patience,
            utilization_threshold=config.utilization_threshold,
            cooldown_epochs=config.cooldown_epochs
        )

        # === 输出头 ===
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

        # === 统计 ===
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.float32))
        self.growth_log: List[str] = []

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq] token ids
            targets: [batch, seq] 目标 token ids（用于 loss 计算）
            mask: [batch, seq] loss 掩码

        Returns:
            logits: [batch, seq, vocab_size]
            info: 字典包含 loss, load_balance_loss, novelty_score 等
        """
        B, T = x.shape
        device = x.device

        # 1. Token Embedding
        emb = self.embedding(x)  # [B, T, d_model]

        # 2. 所有柱计算输出
        all_cols = self.column_pool.all_columns
        if not all_cols:
            # 如果没有柱（极端情况），直通
            logits = self.output_head(emb)
            return logits, {'loss': torch.tensor(0.0), 'lb_loss': torch.tensor(0.0)}

        col_outputs = []
        col_feedbacks = []

        for col in all_cols:
            # 因果掩码（防止模型偷看未来token）
            causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            out, fb = col(emb, causal_mask)
            col_outputs.append(out)
            col_feedbacks.append(fb)

        # 3. 门控路由
        combined, lb_loss, topk_indices = self.corpus_gate(
            emb, col_outputs, col_feedbacks, training=self.training
        )

        # 4. 输出头
        logits = self.output_head(combined)

        # 5. 计算 loss
        loss = torch.tensor(0.0, device=device)
        if targets is not None:
            if mask is not None:
                # 只对掩码位置计算 loss
                active = mask.view(-1) == 1
                if active.any():
                    logits_flat = logits.view(-1, self.cfg.vocab_size)[active]
                    targets_flat = targets.view(-1)[active]
                    loss = F.cross_entropy(logits_flat, targets_flat)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.cfg.vocab_size),
                    targets.view(-1)
                )

        total_loss = loss + self.cfg.load_balance_alpha * lb_loss

        # 6. 海马体处理
        task_loss_val = loss.item()
        encoded, novelty, should_cache = self.hippocampus(emb, task_loss_val)

        info = {
            'loss': total_loss,
            'task_loss': loss,
            'lb_loss': lb_loss,
            'novelty_score': novelty,
            'should_cache': should_cache,
            'num_columns': len(all_cols),
            'num_neuroblasts': len(self.column_pool.neuroblasts),
        }

        return logits, info

    def consolidate(self, optimizer: torch.optim.Optimizer) -> float:
        """
        巩固阶段：海马体蒸馏 → 皮质柱
        在每个 consolidation_interval epoch 后调用
        """
        d_loss = self.hippocampus.consolidate(
            self.column_pool.all_columns,
            self.corpus_gate,
            optimizer,
            batch_size=16,
            epochs=2
        )
        return d_loss

    def check_growth(self, epoch: int, val_acc: float = 0.0) -> List[str]:
        """
        检查并执行生长

        1. grow() 所有 Neuroblast 一步
        2. should_spark() 检查触发条件
        3. 如果触发且未达到最大柱数 → spark()

        Returns: 日志列表
        """
        logs = []

        # 1. 渐进生长
        grow_logs = self.column_pool.grow_all()
        logs.extend(grow_logs)

        # 2. PE 探针 + 利用率 → 触发决策
        utilization = self.corpus_gate.get_expert_utilization(
            self.column_pool.active_count
        )

        should, trigger_info = self.trigger.should_spark(
            self.column_pool.all_columns, utilization, epoch, val_acc
        )
        logs.append(self.trigger.get_summary())

        # 3. 触发 spark
        if should:
            if self.column_pool.active_count < self.cfg.max_columns:
                nb = self.germinal_pool.spark(noise=0.05)
                self.column_pool.add_neuroblast(nb)
                self.trigger.on_spark()
                logs.append(f"  🔥🔥🔥 SPARK! 生成新 Neuroblast (共 {self.column_pool.active_count} 柱)")
            else:
                logs.append(f"  ⚠️ 已达最大柱数 {self.cfg.max_columns}，跳过 spark()")

        # 4. epoch 结束
        self.trigger.on_epoch_end()

        # 5. 更新 PE 探针缓存
        for col in self.column_pool.all_columns:
            if hasattr(col, 'update_pe_prev'):
                col.update_pe_prev()

        self.growth_log.extend(logs)
        return logs

    def get_best_column(self) -> SparkColumn:
        """返回池中最好的柱（用于更新干细胞模板）"""
        mature = self.column_pool.columns
        if not mature:
            return self.column_pool.all_columns[0]
        return mature[0]  # 简化：返回第一个成熟柱

    def update_germinal_template(self):
        """用最好柱更新干细胞模板"""
        best = self.get_best_column()
        self.germinal_pool.update_from_best(best)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        utilization = self.corpus_gate.get_expert_utilization(
            self.column_pool.active_count
        )
        return {
            'epoch': int(self.epoch.item()),
            'num_columns': self.column_pool.active_count,
            'mature_columns': self.column_pool.mature_count,
            'neuroblasts': len(self.column_pool.neuroblasts),
            'novelty_rate': self.hippocampus.get_novelty_rate(),
            'buffer_size': len(self.hippocampus.buffer),
            'expert_utilization': utilization.tolist(),
            'total_sparks': self.germinal_pool.total_sparks,
        }


# ═══════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    from config import ARITHMETIC_CONFIG

    cfg = ARITHMETIC_CONFIG
    print("=== GerminalSpark 初始化测试 ===")

    model = GerminalSpark(cfg)
    print(f"初始柱数: {model.column_pool.active_count}")
    print(f"成熟柱: {model.column_pool.mature_count}")
    print(f"Neuroblast: {len(model.column_pool.neuroblasts)}")

    # 前向传播测试
    x = torch.randint(1, 10, (4, 10))  # [batch, seq]
    targets = torch.randint(1, 10, (4, 10))

    logits, info = model(x, targets)
    print(f"\n前向传播:")
    print(f"  logits: {logits.shape}")
    print(f"  loss: {info['loss'].item():.4f}")
    print(f"  novelty: {info['novelty_score']:.4f}")
    print(f"  柱数: {info['num_columns']}")

    # 检查增长
    print(f"\n增长检查 (epoch 0):")
    logs = model.check_growth(epoch=0)
    for log in logs:
        print(log)

    stats = model.get_stats()
    print(f"\n统计: {stats}")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n总参数: {total_p:,}")
