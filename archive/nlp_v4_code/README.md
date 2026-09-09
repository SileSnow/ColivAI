# V4 Baseline — 字符级英文语言模型

## 实验目标

在 ~5M 参数规模下，建立 Decoder-only Transformer 字符级 LM 基线，
为后续混合 PE / Alibi 实验提供对照。

## 架构

| 参数 | 值 |
|---|---|
| 词表 | BPE 子词, 2000 |
| 层数 | 6 |
| 头数 | 4 (d_head=64) |
| d_model | 256 |
| d_ff | 1024 |
| 位置编码 | sin/cos (冻结) |
| Dropout | 0.0 |
| 参数量 | ~5.24M |

## 文件

```
nlp_v4/
├── config.py           # 超参数集中管理
├── prepare_data.py     # 下载 WikiText-2 + 训练 BPE + 编码分块
├── model.py            # Decoder-only Transformer 定义
├── train.py            # 训练循环
├── eval.py             # PPL 评估 + 文本生成演示
└── README.md           # 本文件
```

## 运行步骤

```bash
# 1. 安装依赖
pip install torch sentencepiece

# 2. 准备数据（需要网络）
python prepare_data.py

# 3. 训练（CPU 约 3~5 小时）
python train.py

# 4. 评估
python eval.py --mode both
```

## 预期结果

- Val PPL: 15~30
- 生成文本有基本英文单词雏形
