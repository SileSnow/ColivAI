"""
SparkTrigger — 生长触发器

双信号决策：
├── monitor_pe()       — PE 探针（柱内部容量饱和检测）
├── monitor_util()     — 柱利用率（门控侧需求检测）
└── should_spark()     — 双信号 AND → spark()
"""

import torch
from typing import List, Tuple


class SparkTrigger:
    """
    生长触发器——决定"要不要生一个新柱"

    双信号 AND 逻辑：
    1. PE 探针：柱的 tanh/sech² 参数变化率 < 阈值 → 容量饱和
    2. 利用率：某柱利用率 > 阈值 → 过载

    两个信号同时亮 → should_spark() = True
    """

    def __init__(self,
                 pe_dormancy_threshold: float = 0.05,
                 pe_patience: int = 3,
                 utilization_threshold: float = 0.7,
                 cooldown_epochs: int = 3,
                 min_epochs_before_spark: int = 5,
                 val_acc_patience: int = 8):
        """
        Args:
            pe_dormancy_threshold: PE 变化率低于此值 → PE 休眠
            pe_patience: 持续 N epoch 休眠才触发
            utilization_threshold: 柱利用率超过此值 → 过载
            cooldown_epochs: spark() 后冷却期（epoch 数）
            min_epochs_before_spark: 前 N epoch 强制不触发（让模型先学）
            val_acc_patience: 连续 N epoch val_acc 不提升 → 备用触发
        """
        self.pe_threshold = pe_dormancy_threshold
        self.pe_patience = pe_patience
        self.util_threshold = utilization_threshold
        self.cooldown = cooldown_epochs
        self.min_epochs = min_epochs_before_spark
        self.val_patience = val_acc_patience

        # 状态
        self.epochs_since_spark = cooldown_epochs + 1
        self.pe_dormant_epochs = 0
        self.best_val_acc = 0.0
        self.val_stagnant_epochs = 0
        self.trigger_history: List[dict] = []

    def monitor_pe(self, columns: List) -> float:
        """
        监控所有柱的 PE 活跃度
        Returns: 平均 PE 活跃度
        """
        activities = []
        for col in columns:
            if hasattr(col, 'get_pe_activity'):
                activities.append(col.get_pe_activity())

        if not activities:
            return 1.0

        avg_activity = sum(activities) / len(activities)
        return avg_activity

    def monitor_util(self, gate_utilization: torch.Tensor) -> Tuple[float, bool]:
        """
        监控柱利用率
        Returns: (最大利用率, 是否过载)
        """
        if gate_utilization.numel() == 0:
            return 0.0, False

        max_util = gate_utilization.max().item()
        is_overloaded = max_util > self.util_threshold
        return max_util, is_overloaded

    def should_spark(self, columns: List,
                     gate_utilization: torch.Tensor,
                     epoch: int = 0,
                     val_acc: float = 0.0) -> Tuple[bool, dict]:
        """
        双信号 AND 决策 + 备用触发

        Returns:
            should: 是否触发 spark()
            info: 判决详情（用于日志/可视化）
        """
        info = {
            'epoch': epoch,
            'pe_activity': 0.0,
            'pe_dormant': False,
            'max_utilization': 0.0,
            'util_overloaded': False,
            'in_cooldown': self.epochs_since_spark < self.cooldown,
            'too_early': epoch < self.min_epochs,
            'val_stagnant': False,
            'decision': False,
        }

        # 0. 最小 epoch 保护
        if epoch < self.min_epochs:
            info['decision'] = False
            self.trigger_history.append(info)
            return False, info

        # 冷却期检查
        if self.epochs_since_spark < self.cooldown:
            info['decision'] = False
            self.trigger_history.append(info)
            return False, info

        # 信号 1：PE 探针
        pe_activity = self.monitor_pe(columns)
        info['pe_activity'] = pe_activity

        if pe_activity < self.pe_threshold:
            self.pe_dormant_epochs += 1
        else:
            self.pe_dormant_epochs = 0

        pe_dormant = self.pe_dormant_epochs >= self.pe_patience
        info['pe_dormant'] = pe_dormant

        # 信号 2：利用率
        max_util, overloaded = self.monitor_util(gate_utilization)
        info['max_utilization'] = max_util
        info['util_overloaded'] = overloaded

        # 备用触发：val_acc 连续不提升
        if val_acc > self.best_val_acc + 0.001:
            self.best_val_acc = val_acc
            self.val_stagnant_epochs = 0
        else:
            self.val_stagnant_epochs += 1
        info['val_stagnant'] = self.val_stagnant_epochs >= self.val_patience

        # 决策：主信号 AND 或 备用信号
        info['decision'] = (pe_dormant and overloaded) or info['val_stagnant']

        self.trigger_history.append(info)
        return info['decision'], info

    def on_spark(self):
        """spark() 被调用后重置状态"""
        self.epochs_since_spark = 0
        self.pe_dormant_epochs = 0
        self.val_stagnant_epochs = 0

    def on_epoch_end(self):
        """每个 epoch 结束时调用"""
        self.epochs_since_spark += 1

    def get_summary(self) -> str:
        """日志摘要"""
        if not self.trigger_history:
            return "SparkTrigger: 无历史"
        last = self.trigger_history[-1]
        parts = [f"PE={last['pe_activity']:.4f}"]
        if last['pe_dormant']: parts.append("休眠")
        parts.append(f"Util={last['max_utilization']:.2%}")
        if last['too_early']: parts.append("太早")
        if last['in_cooldown']: parts.append("冷却")
        if last['val_stagnant']: parts.append("停滞!")
        trigger = "🔥" if last['decision'] else "-"
        return f"SparkTrigger[e{last['epoch']}]: {', '.join(parts)} → {trigger}"


# ═══════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    trigger = SparkTrigger()

    # 模拟：无实际柱，只看逻辑
    print("=== SparkTrigger 测试 ===")

    # 模拟几个 epoch
    util = torch.tensor([0.5, 0.8, 0.3])
    for ep in range(10):
        # 无实际 columns，PE=0（模拟休眠）
        should, info = trigger.should_spark([], util, epoch=ep)
        print(f"Epoch {ep}: {trigger.get_summary()}")
        if should:
            trigger.on_spark()
        trigger.on_epoch_end()
