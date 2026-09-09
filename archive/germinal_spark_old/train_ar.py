"""Germinal Spark — 训练 + 自回归评估"""
import torch, torch.nn as nn, torch.optim as optim, random, sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from train import generate_chain, collate_batch, VOCAB, compute_acc; ID2CHAR = {v:k for k,v in VOCAB.items()}
from config import ARITHMETIC_CONFIG
from germinal_spark import GerminalSpark

@torch.no_grad()
def ar_eval(model, data, n=200):
    """自回归准确率：给 a+b= 让模型自己生成答案"""
    model.eval()
    correct = 0
    for tokens in data[:n]:
        eqs = [j for j,t in enumerate(tokens) if t==VOCAB['=']]
        if len(eqs)<1: continue
        last_eq = eqs[-1]
        prefix = tokens[:last_eq+1]
        target = ''.join(ID2CHAR[t] for t in tokens[last_eq+1:] if t!=VOCAB['EOS'] and t!=VOCAB['\n'])
        gen = list(prefix)
        for _ in range(10):
            x = torch.tensor([gen[-128:]], dtype=torch.long)
            logits, _ = model(x)
            nxt = logits[0,-1].argmax().item()
            gen.append(nxt)
            if nxt==VOCAB['EOS']: break
        pred = ''.join(ID2CHAR[t] for t in gen[len(prefix):] if t!=VOCAB['EOS'] and t!=VOCAB['\n'])
        if pred==target: correct+=1
    return correct/n

torch.manual_seed(42); random.seed(42)
cfg = ARITHMETIC_CONFIG
cfg.initial_columns=1; cfg.max_columns=6; cfg.max_epochs=30
cfg.batch_size=16; cfg.lr=5e-4; cfg.consolidation_interval=3

print('=== Germinal Spark 训练 + 自回归评估 ===')
print(f'初始柱={cfg.initial_columns} max={cfg.max_columns} epochs={cfg.max_epochs}')

train_data = generate_chain(20000, seed=42)
val_data   = generate_chain(2000, seed=999)

tl = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
    collate_fn=lambda b: collate_batch(b, 128, VOCAB['PAD']))
vl = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False,
    collate_fn=lambda b: collate_batch(b, 128, VOCAB['PAD']))

model = GerminalSpark(cfg)
opt = optim.AdamW(model.parameters(), lr=cfg.lr)
sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)

best_ar = 0
for ep in range(cfg.max_epochs):
    # --- train ---
    model.train()
    tl_sum,ta_sum,lb_sum,n = 0,0,0,0
    for x,t,m in tl:
        opt.zero_grad()
        logits,info = model(x,t,m)
        info['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        tl_sum+=info['loss'].item(); lb_sum+=info['lb_loss'].item()
        ta_sum+=compute_acc(logits,t,m); n+=1
    sch.step()
    
    # --- val (teacher-forcing) ---
    model.eval()
    va_sum,vl_sum,vn = 0,0,0
    with torch.no_grad():
        for x,t,m in vl:
            logits,info=model(x,t,m)
            va_sum+=compute_acc(logits,t,m); vl_sum+=info['task_loss'].item(); vn+=1
    
    # --- val (autoregressive) ---
    ar_acc = ar_eval(model, val_data, n=200)
    best_ar = max(best_ar, ar_acc)
    
    s = model.get_stats()
    print(f'E{ep:2d} [{s["num_columns"]}c/{s["mature_columns"]}m] '
          f'loss={tl_sum/n:.3f} ta={ta_sum/n:.2%} | '
          f'va={va_sum/vn:.2%} vl={vl_sum/vn:.3f} | '
          f'AR={ar_acc:.2%}', flush=True)
    
    logs = model.check_growth(epoch=ep)
    for l in logs:
        if any(k in l for k in ['SPARK','成熟','🌱']): print(f'  {l}', flush=True)
    
    if ep>0 and ep%cfg.consolidation_interval==0:
        dl = model.consolidate(opt)
        if dl>0: print(f'  🧠 dl={dl:.4f}', flush=True)
    
    for col in model.column_pool.all_columns:
        if hasattr(col,'update_pe_prev'): col.update_pe_prev()

s = model.get_stats()
print(f'\nDone | best_teacher={va_sum/vn:.2%} best_AR={best_ar:.2%} | cols={s["num_columns"]}/{s["mature_columns"]}m sparks={s["total_sparks"]}')
