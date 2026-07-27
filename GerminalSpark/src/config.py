"""
Germinal Spark — 全局配置
支持算术任务和 NLP 任务双模式切换
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class SparkConfig:
    """Germinal Spark 完整配置"""

    # === 任务选择 ===
    task: Literal["arithmetic", "nlp_wikitext2"] = "arithmetic"

    # === 模型参数 ===
    d_model: int = 64
    d_ff: int = 256
    n_heads: int = 4
    d_head: int = 16          # d_model // n_heads
    max_seq_len: int = 128

    # === 皮质柱 ===
    initial_columns: int = 2        # 初始柱数
    max_columns: int = 5            # 最大柱数（防止无限增长）
    layers_per_column: int = 4      # L4, L23, L5, L6
    growth_order: list = field(default_factory=lambda: ["L6", "L5", "L4", "L23"])
    # 渐进生长顺序：由内向外，深层先生成

    # === GerminalPool ===
    germ_cells: int = 2             # 干细胞数量
    mother_embedding_frozen: bool = True  # 母 embedding 是否冻结

    # === CorpusGate ===
    top_k: int = 2                  # Top-k 路由
    noise_std: float = 0.1          # 噪声门控标准差
    load_balance_alpha: float = 0.01  # 负载均衡 loss 权重

    # === HippocampalHub ===
    hippo_lr: float = 1e-3          # 海马体学习率（高于主模型）
    episodic_buffer_size: int = 256 # 情景缓存容量
    consolidation_interval: int = 5 # 每隔 N epoch 巩固一次

    # === SparkTrigger ===
    pe_dormancy_threshold: float = 0.05   # PE 参数变化率低于此值 → 休眠
    pe_dormancy_patience: int = 3          # 持续 N epoch 休眠才触发
    cooldown_epochs: int = 3              # spark() 后冷却期
    utilization_threshold: float = 0.7    # 柱利用率超过此值 → 过载
    growth_epoch_interval: int = 5        # grow() 调度间隔

    # === 训练 ===
    lr: float = 1e-3
    warmup_epochs: int = 2
    max_epochs: int = 50
    batch_size: int = 32
    early_stop_acc: float = 0.99
    seed: int = 42

    def __post_init__(self):
        """自动推导的配置"""
        self.d_head = self.d_model // self.n_heads

        # 根据任务覆盖部分配置
        if self.task == "arithmetic":
            self.vocab_size = 17      # 0-9, +, -, =, \n, PAD, BOS, EOS
            self.max_seq_len = 128
        elif self.task == "nlp_wikitext2":
            self.vocab_size = 2000     # BPE 2000
            self.max_seq_len = 256
            self.lr = 3e-4
            self.batch_size = 16
            self.early_stop_acc = None  # NLP 不用 Acc 早停

    @property
    def column_params(self) -> int:
        """单个 SparkColumn 的参数量（估算）"""
        d = self.d_model
        ff = self.d_ff
        # L4_Input + L23_Assoc(attn) + L5_Output(ffn) + L6_Feedback + PE
        l4 = d * d
        l23 = 3 * d * d + d * d + self.n_heads * 3  # QKV + O + PE
        l5 = d * ff + ff * d
        l6 = d * d
        return l4 + l23 + l5 + l6

    @property
    def total_params(self) -> int:
        """总参数量估算"""
        emb = self.vocab_size * self.d_model
        cols = self.initial_columns * self.column_params
        gate = self.d_model * self.initial_columns
        hippo = self.d_model * self.d_model * 3  # DG + novelty + distill
        return emb + cols + gate + hippo


# === 预设配置 ===

ARITHMETIC_CONFIG = SparkConfig(task="arithmetic")

NLP_WIKITEXT2_CONFIG = SparkConfig(task="nlp_wikitext2")


if __name__ == "__main__":
    cfg = ARITHMETIC_CONFIG
    print(f"=== {cfg.task} 配置 ===")
    print(f"d_model={cfg.d_model}, d_ff={cfg.d_ff}, n_heads={cfg.n_heads}")
    print(f"vocab_size={cfg.vocab_size}, max_seq_len={cfg.max_seq_len}")
    print(f"初始柱数={cfg.initial_columns}, 最大={cfg.max_columns}")
    print(f"单柱参数: {cfg.column_params:,}")
    print(f"总参数估计: {cfg.total_params:,}")
    print(f"\n增长顺序: {' → '.join(cfg.growth_order)}")
    print(f"PE休眠阈值: {cfg.pe_dormancy_threshold}, 冷却期: {cfg.cooldown_epochs} epoch")
