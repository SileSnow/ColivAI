# 小模型位置编码与嵌入策略消融实验 — 项目总结

**研究员**: 西里西亚的雪  
**时间**: 2026年7月  
**服务器**: 49.232.189.241, 4核Xeon, 15GB RAM, Ubuntu 24.04  
**虚拟环境**: /root/airesearch/.torchvenv

---

## 第一阶段：算术推理（/root/airesearch/）

800K 参数微型 Transformer，一阶±运算 + 连续链算术。

| 实验 | PE | 核心发现 |
|---|---|---|
| baseline | sin/cos | ValAcc 99.0% |
| verify3.1 | 解耦合预训练 Emb | 38K模型: 99% vs 78%（打破天花板） |
| verify3.2 | tanh/sech²（1步） | tanh全休眠（0/16激活） |
| verify3.3 | tanh/sech²（连续链） | tanh全休眠 |
| verify3.3.1 | tanh/sech²（非零init） | tanh仅~1%微动 |

**结论**: 800K规模下 tanh/sech² 不学习。

---

## 第二阶段：NLP 语言模型（/root/nlp_v4/）

Decoder-only Transformer，WikiText 英文 LM。

### 最终实验矩阵

| 实验 | PE | 架构 | 数据 | 词表 | Val PPL | 备注 |
|---|---|---|---|---|---|---|
| V4-Baseline | sin/cos | 经典 | WikiText-2 | 2000 | 25.14 | Epoch 12 |
| V4-ALiBi | ALiBi | 经典 | WikiText-2 | 2000 | 24.73 | 收敛快3epoch |
| V4-Mixed | tanh/sech² | 经典 | WikiText-2 | 2000 | 22.45 | **tanh首次学会!** |
| V4-RoPE | RoPE | LLaMA | WikiText-103 | 5000 | **10.31** 🏆 | |
| V4-Mixed-103 | tanh/sech² | 经典 | WikiText-103 | 5000 | 10.44 | |
| V4-Mixed-LLaMA-103 | tanh/sech² | LLaMA | WikiText-103 | 5000 | 10.69 | |

### WikiText-103 三模型对比（同数据、同词表）

| 模型 | PE | 架构 | Val PPL |
|---|---|---|---|
| **RoPE** | RoPE（固定旋转） | LLaMA | **10.31** 🏆 |
| Mixed-103 | tanh/sech²（可学习） | 经典 | 10.44 |
| Mixed-LLaMA | tanh/sech²（可学习） | LLaMA | 10.69 |

### 核心结论

1. **RoPE 最优**: 固定旋转编码在小模型上也领先
2. **tanh/sech² 紧追**: 大数据（WikiText-103）上仅差1.3%，证明了「可学习PE」的价值
3. **规模阈值被找到**: tanh 在 800K 下不学，在 5M+512长度下大幅学习
4. **LLaMA架构在6M下无优势**: Pre-RMSNorm 反而削弱了 tanh 梯度流
5. **数据 > 架构**: WikiText-103 让 PPL 从 22→10，远超任何架构改进
6. **PPL ≠ 生成质量**: PPL 10 仍不能流畅输出，生成需要 30M+ 参数
7. **ALiBi 收敛最快**: 低资源场景的实用选择

---

## 模型文件

| 文件 | 模型 | Val PPL |
|---|---|---|
| /root/nlp_v4/models/epoch10_ppl25.55.pt | Baseline (sin/cos) | 25.55 |
| /root/nlp_v4/models/mixed_epoch13_ppl22.45.pt | Mixed (WikiText-2) | 22.45 |
| /root/nlp_v4/models/best_rope.pt | RoPE | 10.31 |
| /root/nlp_v4/models/best_mixed103.pt | Mixed-103 | 10.44 |
| /root/nlp_v4/models/best_mixed_llama103.pt | Mixed-LLaMA-103 | 10.69 |

---

## 设计教训

1. **tanh 初始化不能为0**: w=0,v=0 导致梯度死锁
2. **sech² 的 cosh 需 clamp**: d>85 时 float32 溢出
3. **Python 输出需 -u**: nohup 重定向时默认缓冲
4. **S3 下载源已死**: 需用 HF_ENDPOINT=https://hf-mirror.com
5. **PPL 跨词表不可比**: 大词表天然更低，需同词表对比
6. **小模型生成质量天花板**: 6M 参数 PPL 10 仍不流畅，这是容量问题不是 PE 问题
