"""
Germinal Spark — 训练脚本

模式：
  M0: baseline    — 固定柱，不增长
  M1: gs_full     — 渐进生长 + 海马体蒸馏
  M2: gs_net2net  — Net2Net 瞬间复制（消融 A）
  M3: gs_nodistill— 无海马体蒸馏（消融 B）

用法：
  python3 train.py --mode gs_full --epochs 30 --chain   # 连续链算术
  python3 train.py --mve                                  # 跑全部消融
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse, random, os, sys

# 确保可以从上级目录（GerminalSpark/）也能正常运行
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(__file__))

from config import SparkConfig, ARITHMETIC_CONFIG
from germinal_spark import GerminalSpark

VOCAB = {
    '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
    '+':10,'-':11,'=':12,'\n':13,'PAD':14,'BOS':15,'EOS':16,
}

def encode_arithmetic(expr: str) -> list:
    tokens = []
    for ch in expr:
        if ch in VOCAB: tokens.append(VOCAB[ch])
    return tokens

def generate_arithmetic(num_samples=20000, seed=42):
    """1步算术"""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        a, b = random.randint(0,99), random.randint(0,99)
        op = random.choice(['+','-'])
        if op=='-': a,b=max(a,b),min(a,b)
        c = a+b if op=='+' else a-b
        tokens = [VOCAB['BOS']]+encode_arithmetic(f"{a}{op}{b}={c}")+[VOCAB['EOS']]
        data.append(tokens)
    return data

def generate_chain(num_samples=20000, min_steps=2, max_steps=8, seed=42):
    """连续链算术"""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        steps = random.randint(min_steps, max_steps)
        a = random.randint(1,99)
        lines = []
        for s in range(steps):
            op = random.choice(['+','-'])
            b = random.randint(1,99)
            if op=='-': a,b=max(a,b),min(a,b); result=a-b
            else: result=a+b
            lines.append(f"{a}{op}{b}={result}")
            a = result
        full = '\n'.join(lines)+f'\n={result}'
        tokens = [VOCAB['BOS']]+encode_arithmetic(full)+[VOCAB['EOS']]
        data.append(tokens)
    return data

def collate_batch(batch, max_len=128, pad_id=14):
    batch = [b[:max_len] for b in batch]
    max_l = max(len(b) for b in batch)
    x = torch.full((len(batch), max_l), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), max_l)
    for i, seq in enumerate(batch):
        x[i,:len(seq)] = torch.tensor(seq, dtype=torch.long)
        # Mask: 最后一个 '=' 之后
        eq_pos = None
        for j in range(len(seq)-1, -1, -1):
            if seq[j]==VOCAB['=']:
                eq_pos=j; break
        if eq_pos is not None:
            mask[i, eq_pos:len(seq)] = 1.0
    # Shifted targets: position i predicts token i+1
    targets = torch.full_like(x, pad_id)
    targets[:, :-1] = x[:, 1:].clone()
    targets[:, -1] = VOCAB['EOS']
    return x, targets, mask

def compute_acc(logits, targets, mask):
    active = mask.view(-1)==1
    if not active.any(): return 0.0
    preds = logits.view(-1, logits.shape[-1])[active].argmax(-1)
    return (preds==targets.view(-1)[active]).float().mean().item()

def train_epoch(model, loader, opt, epoch):
    model.train()
    tot_loss, tot_acc, tot_lb, n = 0,0,0,0
    for x, targets, mask in loader:
        opt.zero_grad()
        logits, info = model(x, targets, mask)
        loss = info['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss+=loss.item(); tot_acc+=compute_acc(logits,targets,mask)
        tot_lb+=info['lb_loss'].item(); n+=1
    return {'loss':tot_loss/n,'acc':tot_acc/n,'lb':tot_lb/n}

@torch.no_grad()
def validate(model, loader):
    model.eval()
    tot_acc, tot_loss, n = 0,0,0
    for x, targets, mask in loader:
        logits, info = model(x, targets, mask)
        tot_acc+=compute_acc(logits,targets,mask)
        tot_loss+=info['task_loss'].item(); n+=1
    return tot_acc/n, tot_loss/n

def train(cfg: SparkConfig, mode="gs_full", use_chain=True):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"  Germinal Spark | mode={mode} | chain={'yes' if use_chain else 'no'}")
    print(f"  init_cols={cfg.initial_columns} max_cols={cfg.max_columns}")
    print(f"{'='*60}\n")

    # 数据
    gen = generate_chain if use_chain else generate_arithmetic
    train_data = gen(20000, seed=cfg.seed)
    val_data   = gen(2000, seed=cfg.seed+999)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, cfg.max_seq_len, VOCAB['PAD']))
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, cfg.max_seq_len, VOCAB['PAD']))

    # 模型
    model = GerminalSpark(cfg)

    # Mode 调整
    if mode=="baseline":
        model.trigger.pe_patience = 99999
        model.trigger.util_threshold = 99
    elif mode=="gs_net2net":
        model.growth_style = "net2net"
    elif mode=="gs_nodistill":
        model.consolidation_enabled = False

    model.train()
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    best_val_acc = 0.0
    history = []

    for ep in range(cfg.max_epochs):
        tm = train_epoch(model, train_loader, opt, ep)
        va, vl = validate(model, val_loader)
        sch.step()

        best_val_acc = max(best_val_acc, va)

        stats = model.get_stats()
        ci = f"[{stats['num_columns']}c/{stats['mature_columns']}m]"
        print(f"E{ep:3d} {ci} train_loss={tm['loss']:.3f} acc={tm['acc']:.2%} | val_acc={va:.2%}")

        if mode!="baseline":
            logs = model.check_growth(epoch=ep)
            for l in logs:
                if any(k in l for k in ['SPARK','成熟','🌱']): print(f"  {l}")

        if mode in ("gs_full","gs_net2net"):
            if ep>0 and ep%cfg.consolidation_interval==0:
                dl = model.consolidate(opt)
                if dl>0: print(f"  🧠 distill_loss={dl:.4f}")

        for col in model.column_pool.all_columns:
            if hasattr(col,'update_pe_prev'): col.update_pe_prev()

        if cfg.early_stop_acc and va>=cfg.early_stop_acc:
            print(f"\n🎉 早停 ValAcc={va:.2%}")
            break

    print(f"\nDone | best_acc={best_val_acc:.2%} | cols={model.column_pool.active_count} | sparks={model.germinal_pool.total_sparks}")
    return model, history


def run_mve(cfg: SparkConfig, use_chain=True):
    """MVE 四轮消融"""
    experiments = {
        "M0_baseline":   "baseline",
        "M1_gs_full":    "gs_full",
        "M2_net2net":    "gs_net2net",
        "M3_nodistill":  "gs_nodistill",
    }
    results = {}
    for name, mode in experiments.items():
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        try:
            m, h = train(cfg, mode=mode, use_chain=use_chain)
            s = m.get_stats()
            results[name] = {
                'best_acc': max(e['val_acc'] for e in h) if h else 0,
                'final_cols': s['num_columns'],
                'mature_cols': s['mature_columns'],
                'sparks': s['total_sparks'],
            }
        except Exception as e:
            print(f"  ❌ {e}")
            results[name] = {'error': str(e)}

    print(f"\n{'='*60}\n  MVE Summary\n{'='*60}")
    for n, r in results.items():
        if 'error' in r: print(f"  {n}: ❌ {r['error']}")
        else: print(f"  {n}: acc={r['best_acc']:.2%} cols={r['final_cols']}/{r['mature_cols']}m sparks={r['sparks']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="gs_full",
                        choices=["baseline","gs_full","gs_net2net","gs_nodistill"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--chain", action="store_true", default=True)
    parser.add_argument("--no-chain", action="store_true")
    parser.add_argument("--mve", action="store_true")
    args = parser.parse_args()

    cfg = ARITHMETIC_CONFIG
    cfg.max_epochs = args.epochs
    cfg.initial_columns = 1  # MVE: 从1柱开始
    cfg.max_columns = 6
    cfg.batch_size = 16
    cfg.lr = 5e-4
    cfg.early_stop_acc = None  # 不早停，给增长留空间
    cfg.consolidation_interval = 3

    use_chain = args.chain and not args.no_chain

    if args.mve:
        run_mve(cfg, use_chain=use_chain)
    else:
        train(cfg, mode=args.mode, use_chain=use_chain)

# ═══════════════════════════════════════
# 自回归评估
# ═══════════════════════════════════════

@torch.no_grad()
def ar_evaluate(model, data, num_samples=100):
    """自回归评估：给出 a+b= 让模型自己生成答案"""
    model.eval()
    correct = 0
    for i in range(min(num_samples, len(data))):
        tokens = data[i]
        # 找到最后一个 '=' 的位置
        eq_positions = [j for j, t in enumerate(tokens) if t == VOCAB['=']]
        if len(eq_positions) < 1:
            continue
        last_eq = eq_positions[-1]
        # 前缀：到最后一个 '='（含）
        prefix = tokens[:last_eq + 1]
        # 目标答案
        target = tokens[last_eq + 1:]  # 不包括 EOS
        target_str = ''.join(VOCAB[t] for t in target if t != VOCAB['EOS'])
        
        # 自回归生成
        gen = list(prefix)
        with torch.no_grad():
            for _ in range(10):
                x = torch.tensor([gen[-128:]], dtype=torch.long)
                logits, _ = model(x)
                nxt = logits[0, -1].argmax().item()
                gen.append(nxt)
                if nxt == VOCAB['EOS']:
                    break
        
        pred_str = ''.join(VOCAB[t] for t in gen[len(prefix):] if t != VOCAB['EOS'])
        if pred_str == target_str:
            correct += 1
    
    return correct / min(num_samples, len(data))
