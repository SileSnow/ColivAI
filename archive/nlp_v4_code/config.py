"""
V4 Baseline — 字符级英文语言模型 超参数
========================================
所有可调参数集中管理，方便后续实验对比。
"""

import os

# ===== 路径 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===== 数据 =====
VOCAB_SIZE = 2000
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SEQ_LEN = 512
BPE_MODEL_PREFIX = os.path.join(DATA_DIR, f"spm_bpe_{VOCAB_SIZE}")

# ===== 模型 =====
D_MODEL = 256
NHEAD = 4          # 每头 64 维
NLAYER = 6
D_FF = 1024
MAX_LEN = 600      # 最大位置编码长度（> SEQ_LEN 留余量）
DROPOUT = 0.0

# ===== 训练 =====
BATCH_SIZE = 16    # 512 长度，降 batch 防 OOM
EPOCHS = 20
LR = 3e-4
LR_MIN = 1e-5
WARMUP_EPOCHS = 3
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# ===== 评估 =====
EVAL_EVERY_N_BATCHES = 100
GEN_EVERY_N_EPOCHS = 5
GEN_MAX_NEW_TOKENS = 50
GEN_TEMPERATURE = 0.8
