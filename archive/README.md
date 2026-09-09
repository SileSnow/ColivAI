# AI Research Archive

个人 AI 研究实验代码归档 —— 由 `archive` 文件夹整理而来(原始文件夹未做任何改动,本仓库为桌面上的独立副本)。

> **GitHub**: https://github.com/SileSnow (本仓库准备推送到个人 GitHub, 仓库名 `airesearch`)

## 仓库结构

| 目录 | 内容 | 说明 |
|---|---|---|
| [airesearch/](airesearch/) | 算术推理实验 (第一阶段) | ~800K 微型 Transformer, 一阶 ± 运算 + 连续链算术, 位置编码研究 |
| [germinal_spark_old/](germinal_spark_old/) | GerminalSpark 生物启发架构 (旧版存档) | 干细胞池/皮质柱/门控路由 等模块化神经形态研究 |
| [nlp_v4_code/](nlp_v4_code/) | 小模型位置编码消融 (第二阶段) | Decoder-only 字符级 LM, sin/cos vs ALiBi vs tanh/sech² vs RoPE 对比 |

详细进展与结论见 [nlp_v4_code/PROJECT_STATUS.md](nlp_v4_code/PROJECT_STATUS.md)。

## 实验脉络

1. **算术推理阶段** (`airesearch/`): baseline / v1–v3 / verify3.1–3.3.1 各实验目录,
   研究 **tanh/sech² 可学习位置编码在极小模型 (800K) 下的行为**。
   结论: tanh 在 800K 规模下不学习(全休眠), 需要 5M+ 参数才有意义。
2. **NLP 语言模型阶段** (`nlp_v4_code/`): WikiText 上对比多种位置编码,
   RoPE 最优 (Val PPL 10.31), tanh/sech² 在 5M 规模大幅学习 (10.44), 仅差 1.3%。

## 代码约定

各实验目录结构相似, 典型入口:

- `train*.py` — 训练脚本 (每种 PE / 架构一个文件)
- `model*.py` — 模型定义
- `config.py` — 超参数
- `eval.py` / `verify*.py` — 评估
- `datagen.py` / `prepare_data.py` — 数据生成/下载

## 哪些文件不在仓库中?

被 [.gitignore](.gitignore) 排除(不会上传, 但保留在本地副本):

- `.torchvenv/` — Python 虚拟环境 (~1.3 GB, 内含 >100MB 二进制, GitHub 不接收)
- `checkpoints*` / `*.pth` / `*.pt` — 训练检查点
- `*.log` / `*.txt` / `logs/` — 日志与运行输出
- `airesearch/data/` — MNIST 原始数据 (训练时可重新下载)
- `*.png` / `figures/` — 图表

## 推送方法(在 GitHub 网页创建同名空仓库后执行)

```bash
cd C:/Users/Lenovo/Desktop/airesearch
git remote add origin https://github.com/SileSnow/airesearch.git
git branch -M main
git push -u origin main
```

> 仓库不使用 Git LFS, 已入库的均为 GitHub 允许的普通文件 (单文件 < 50 MB)。
