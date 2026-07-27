"""Germinal Spark — Scheduled Sampling + 单步算术 + 自回归评估"""
import torch, torch.nn as nn, torch.optim as optim, random, sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from train import generate_arithmetic, collate_batch, VOCAB, compute_acc
from config import ARITHMETIC_CONFIG
from germinal_spark import GerminalSpark

ID2CHAR = {v:k for k,v in VOCAB.items()}

@torch.no_grad()
def ar_eval(model, data, n=200):
    """自回归准确率"""
    model.eval()
    correct = 0
    for tokens in data[:n]:
        eqs = [j for j,t in enumerate(tokens) if t==VOCAB['=']]
        if len(eqs)<1: continue
        last_eq = eqs[-1]
        prefix = tokens[:last_eq+1]
        target = ''.join(ID2CHAR[t] for t in tokens[last_eq+1:] if t not in [VOCAB['EOS'],VOCAB['\n']])
        gen = list(prefix)
        for _ in range(8):
            x = torch.tensor([gen[-64:]], dtype=torch.long)
            logits, _ = model(x)
            nxt = logits[0,-1].argmax().item()
            gen.append(nxt)
            if nxt==VOCAB['EOS']: break
        pred = ''.join(ID2CHAR[t] for t in gen[len(prefix):] if t not in [VOCAB['EOS'],VOCAB['\n']])
        if pred==target: correct+=1
    return correct/n

torch.manual_seed(42); random.seed(42)
cfg = ARITHMETIC_CONFIG
cfg.initial_columns=1; cfg.max_columns=6; cfg.max_epochs=40
cfg.batch_size=32; cfg.lr=1e-3; cfg.consolidation_interval=3

print(f'=== Germinal Spark — Scheduled Sampling + 单步算术 ===')
print(f'epochs={cfg.max_epochs} batch={cfg.batch_size} lr={cfg.lr}')

train_data = generate_arithmetic(20000, seed=42)
val_data   = generate_arithmetic(2000, seed=999)
print(f'数据: train={len(train_data)} val={len(val_data)}')

tl = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
    collate_fn=lambda b: collate_batch(b, 32, VOCAB['PAD']))
vl = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False,
    collate_fn=lambda b: collate_batch(b, 32, VOCAB['PAD']))

model = GerminalSpark(cfg)
opt = optim.AdamW(model.parameters(), lr=cfg.lr)
sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)

best_ar = 0
for ep in range(cfg.max_epochs):
    # Scheduled sampling probability: 0→0.5 over training
    ss_prob = min(0.5, ep / cfg.max_epochs * 0.5) if ep > 5 else 0.0
    
    # --- train ---
    model.train()
    tl_sum,ta_sum,lb_sum,n,ss_count = 0,0,0,0,0
    for x,t,m in tl:
        opt.zero_grad()
        
        # Scheduled sampling: 以概率 ss_prob 用模型自己预测的 token 替代 ground truth
        if ss_prob > 0:
            B, T = x.shape
            # 先做一次前向获取预测
            with torch.no_grad():
                logits_pred, _ = model(x)
                pred_tokens = logits_pred.argmax(dim=-1)
            # 随机决定哪些位置用预测替代
            ss_mask = torch.rand(B, T) < ss_prob
            # 只替换 loss mask 内的位置
            ss_mask = ss_mask & (m.bool())
            if ss_mask.any():
                x_ss = x.clone()
                x_ss[ss_mask] = pred_tokens[ss_mask]
                ss_count += 1
            else:
                x_ss = x
        else:
            x_ss = x
        
        logits,info = model(x_ss, t, m)  # input=SS混合, target=shifted targets
        # 注意：这里 target 应该仍是原始 t，不是 x_ss
        # 但 Scheduled Sampling 用 x_ss 作为输入，target 保持原样
        info['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        tl_sum+=info['loss'].item(); lb_sum+=info['lb_loss'].item()
        ta_sum+=compute_acc(logits,t,m); n+=1
    sch.step()
    
    # --- val (teacher-forcing) ---
    model.eval()
    va_sum,vn = 0,0
    with torch.no_grad():
        for x,t,m in vl:
            logits,info=model(x,t,m)
            va_sum+=compute_acc(logits,t,m); vn+=1
    
    # --- val (autoregressive) ---
    ar_acc = ar_eval(model, val_data, n=200)
    best_ar = max(best_ar, ar_acc)
    
    s = model.get_stats()
    print(f'E{ep:2d} [{s["num_columns"]}c/{s["mature_columns"]}m] '
          f'loss={tl_sum/n:.3f} ta={ta_sum/n:.2%} | '
          f'va={va_sum/vn:.2%} | AR={ar_acc:.2%} '
          f'ss_p={ss_prob:.2f}', flush=True)
    
    logs = model.check_growth(epoch=ep)
    for l in logs:
        if any(k in l for k in ['SPARK','成熟','🌱']): print(f'  {l}', flush=True)
    
    if ep>0 and ep%cfg.consolidation_interval==0:
        dl = model.consolidate(opt)
        if dl>0: print(f'  🧠 dl={dl:.4f}', flush=True)
    
    for col in model.column_pool.all_columns:
        if hasattr(col,'update_pe_prev'): col.update_pe_prev()

s = model.get_stats()
print(f'\nDone | best_AR={best_ar:.2%} | cols={s["num_columns"]}/{s["mature_columns"]}m sparks={s["total_sparks"]}')
