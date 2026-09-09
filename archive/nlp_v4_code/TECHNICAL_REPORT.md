# 小模型位置编码系统性消融：从 sin/cos 到 RoPE

**西里西亚的雪**  
2026 年 7 月 17 日

---

## 摘要

在 5–6M 参数 Decoder-only Transformer 上，对四种位置编码方案进行系统性消融：sin/cos、ALiBi、tanh/sech²、RoPE，覆盖 WikiText-2 (BPE 2000, ~4M tokens) 和 WikiText-103 (BPE 5000, ~25M tokens) 两个数据规模。核心发现：(1) **RoPE 的优势是数据依赖的**——大数据上最优 (Val PPL 10.31)，小数据上反而垫底 (32.24)；(2) **tanh/sech² 在小数据上是王者** (22.45)；(3) 可学习 PE 的规模阈值约在 5M 参数 + 512 序列长度；(4) 提出 arctan+1/(1+d²) 替代方案解决长距离梯度消失；(5) LLaMA 架构在 6M 下无增益。本文为小模型 PE 选择提供了完整实证矩阵。

---

## 1. 引言

Transformer 的自注意力机制对位置不敏感，位置编码 (Position Encoding, PE) 成为关键组件。自 2017 年 sin/cos 提出以来，PE 经历了 ALiBi (2022) 和 RoPE (2023) 两次重要迭代，后者已成为 LLaMA、DeepSeek 等现代 LLM 的标准。

然而，这些方案在 **小模型 (< 10M) + 小数据 (< 100M tokens)** 场景下的比较研究尚不充分。本文在严格控制的计算预算下 (单机 4 核 CPU, 15GB RAM)，进行了完整消融对比。

---

## 2. 实验设置

### 2.1 模型架构

| 参数 | WikiText-2 组 | WikiText-103 组 |
|---|---|---|
| 参数量 | ~5.25M | ~6.02M |
| 层数/头数 | 6/4 | 同 |
| d_model/d_ff | 256/1024 | 同 |
| 词表 | BPE 2000 | BPE 5000 |
| 序列长度 | 256/512 | 512 |
| 训练量 | 8–20 epoch | 2 epoch |
| 优化器 | AdamW, LR=3e-4 | 同 |

### 2.2 四种位置编码

| PE | 原理 | 可学习参数 |
|---|---|---|
| **sin/cos** | 绝对位置向量加到 token embedding | 0 |
| **ALiBi** | 固定线性偏置 bias = -m·\|i-j\| | 0 |
| **tanh/sech²** | w·tanh(v·d) + sech²(τ·d)，每头可学习 | 72 |
| **RoPE** | 旋转 Q,K 向量，内积天然编码相对位置 | 0 |

### 2.3 arctan 变体（本文提出）

为解决 tanh 在长距离下梯度指数消失的问题，提出：

$$\text{bias}(d) = w \cdot \arctan(v \cdot d) + \frac{1}{1 + (\tau \cdot d)^2}$$

梯度为 1/(1+x²)，在长距离保持多项式衰减。1/(1+d²) 是 arctan 的导数，与原版 sech² (= tanh 的导数) 形成数学对偶。

---

## 3. 结果

### 3.1 WikiText-2 小数据组 (BPE 2000, ~4M tokens)

| 模型 | PE | Val PPL |
|---|---|---|
| **Mixed** | **tanh/sech²** | **22.45** 🏆 |
| ALiBi | ALiBi | 24.73 |
| Baseline | sin/cos | 25.14 |
| **RoPE-WT2** | **RoPE** | **32.24** |

### 3.2 WikiText-103 大数据组 (BPE 5000, ~25M tokens)

| 模型 | PE | 架构 | Val PPL |
|---|---|---|---|
| **RoPE** | **RoPE** | LLaMA | **10.31** 🏆 |
| Mixed-103 | tanh/sech² | 经典 | 10.44 |
| Mixed-arctan | arctan+1/(1+d²) | 经典 | 10.51 |
| Mixed-LLaMA | tanh/sech² | LLaMA | 10.69 |

### 3.3 可学习 PE 参数分析

**规模阈值发现**：tanh/sech² 在 800K 参数下不发生学习 (算术任务，参数变化 < 1%)，但在 5M + 512 长度下大幅学习 (参数变化 50–100%)，层间出现功能分化：

| 层 | 行为 | 含义 |
|---|---|---|
| 浅层 (L0–L1) | w↑, v<0, τ↓ | 专注近邻 |
| 深层 (L3–L5) | w↓, v>0, τ↑ | 倾向远距离 |

---

## 4. 讨论

### 4.1 RoPE 的数据依赖性（核心发现）

本文最重要的发现是 **RoPE 的表现高度依赖数据规模**：

```
小数据 (4M tokens):  RoPE 32.24   ← 最差，不如 sin/cos
大数据 (25M tokens): RoPE 10.31   ← 最优
```

RoPE 的多尺度旋转 (128 对维度 × 6 层) 需要足够的数据让 Q,K 投影层学会「把哪些语义放在哪个频率维度」。小数据下校准不充分，多频率反而成为噪声。**在小数据场景，简单的 ALiBi 或可学习的 tanh/sech² 是更优选择。**

### 4.2 可学习 vs 人工设计

大数据上 RoPE (固定) 与 tanh/sech² (可学习) 差距仅 1.3% (10.31 vs 10.44)。小数据上 tanh/sech² 大幅领先 RoPE (22.45 vs 32.24)。**可学习 PE 在数据稀缺时具有明显优势。**

### 4.3 LLaMA 架构在 6M 下无增益

经典架构 + tanh/sech² (10.44) 略优于 LLaMA 架构 + tanh/sech² (10.69)。推测 Pre-RMSNorm 削弱了可学习偏置的梯度流。这些工程优化需更大规模才能体现价值。

### 4.4 效应大小排序

| 因素 | 变化 | PPL 影响 |
|---|---|---|
| **数据量** | 4M → 25M tokens | ↓53% |
| **PE (小数据)** | sin/cos → tanh | ↓11% |
| **PE (大数据)** | sin/cos → RoPE | ↓31%* |
| **架构** | 经典 → LLaMA | ↑2% |

*估计值，因跨词表不可直接比较。

---

## 5. 结论

1. **没有绝对最优的 PE**——RoPE 在大数据上最强，tanh/sech² 在小数据上是王者
2. **RoPE 的优势是数据依赖的**，小数据下 128 对旋转频率反而成了噪声
3. **可学习 PE 在低资源场景更有价值**——更多数据时，固定方案也能追平
4. **tanh/sech² 的规模阈值约在 5M + 512 长度**，低于此无效
5. **arctan 解决了长距离梯度问题**，但软饱和限制了天花板
6. **6M 模型不必追求 LLaMA 架构**，经典 Transformer 足够

---

## 附录：完整实验矩阵

| 实验 | PE | 数据 | 词表 | Val PPL |
|---|---|---|---|---|
| Baseline | sin/cos | WikiText-2 | 2000 | 25.14 |
| ALiBi | ALiBi | WikiText-2 | 2000 | 24.73 |
| Mixed | tanh/sech² | WikiText-2 | 2000 | **22.45** |
| RoPE-WT2 | RoPE | WikiText-2 | 2000 | 32.24 |
| RoPE | RoPE | WikiText-103 | 5000 | **10.31** |
| Mixed-103 | tanh/sech² | WikiText-103 | 5000 | 10.44 |
| arctan | arctan+1/(1+d²) | WikiText-103 | 5000 | 10.51 |
| Mixed-LLaMA | tanh/sech² | WikiText-103 | 5000 | 10.69 |

---

代码与权重: https://github.com/SileSnow/ColivAI  
实验环境: 4×Intel Xeon, 15GB RAM, PyTorch 2.x, Ubuntu 24.04, 无 GPU
