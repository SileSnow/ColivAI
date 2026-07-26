"""V5.1: 生长生命周期实验 — 连续链算术 + PE探针 + SparkTrigger"""
import torch, torch.nn as nn, torch.optim as optim, random, sys
from torch.optim.lr_scheduler import CosineAnnealingLR

V = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
     '+':10,'-':11,'=':12,'\n':13,'PAD':14,'BOS':15,'EOS':16}
I = {v:k for k,v in V.items()}
PAD = V['PAD']; BOS = V['BOS']; EOS = V['EOS']; EQ = V['=']

def enc(t):
    return [V.get(c, PAD) for c in t]

def gen_chain(n=20000, min_s=2, max_s=8, seed=42):
    random.seed(seed); data = []
    for _ in range(n):
        steps = random.randint(min_s, max_s); a = random.randint(1,99)
        lines = []
        for s in range(steps):
            op = random.choice(['+','-']); b = random.randint(1,99)
            if op == '-': a,b = max(a,b),min(a,b); result = a-b
            else: result = a+b
            lines.append(f'{a}{op}{b}={result}'); a = result
        full = '\n'.join(lines) + f'\n={result}'
        tokens = [BOS] + enc(full) + [EOS]
        data.append(tokens)
    return data

def collate(batch, max_len=128):
    batch = [b[:max_len] for b in batch]; L = max(len(b) for b in batch)
    x = torch.full((len(batch), L), PAD, dtype=torch.long)
    m = torch.zeros(len(batch), L)
    for i, seq in enumerate(batch):
        x[i,:len(seq)] = torch.tensor(seq, dtype=torch.long)
        for j in range(len(seq)-1, -1, -1):
            if seq[j] == EQ: m[i, j:len(seq)] = 1.0; break
    t = torch.full_like(x, PAD); t[:,:-1] = x[:,1:].clone(); t[:,-1] = EOS
    return x, t, m

def compute_acc(logits, targets, mask):
    active = mask.view(-1) == 1
    if not active.any(): return 0.0
    return (logits.view(-1, logits.shape[-1])[active].argmax(-1)
            == targets.view(-1)[active]).float().mean().item()

@torch.no_grad()
def ar_eval(model, data, n=100):
    model.eval(); correct = 0
    for tokens in data[:n]:
        eqs = [j for j,t in enumerate(tokens) if t == EQ]
        if not eqs: continue
        prefix = tokens[:eqs[-1]+1]
        target = ''.join(I[t] for t in tokens[eqs[-1]+1:] if t not in [EOS, V['\n']])
        gen = list(prefix)
        for _ in range(10):
            x = torch.tensor([gen[-128:]], dtype=torch.long)
            logits, _ = model(x)
            nxt = logits[0,-1].argmax().item(); gen.append(nxt)
            if nxt == EOS: break
        pred = ''.join(I[t] for t in gen[len(prefix):] if t not in [EOS, V['\n']])
        if pred == target: correct += 1
    return correct / n

from config import ARITHMETIC_CONFIG
from germinal_spark import GerminalSpark

torch.manual_seed(42); random.seed(42)
cfg = ARITHMETIC_CONFIG
cfg.vocab_size = 17; cfg.initial_columns = 1; cfg.max_columns = 4
cfg.d_model = 128; cfg.d_ff = 512; cfg.n_heads = 4
cfg.max_epochs = 30; cfg.batch_size = 16; cfg.lr = 3e-4
cfg.max_seq_len = 128; cfg.consolidation_interval = 999
cfg.warmup_epochs = 0

# 加载解耦合 embedding (vocab=17, dim=128 — 需投影)
pt_emb = torch.load('/root/airesearch/verify3.1/embedding.pth', map_location='cpu', weights_only=False)
# verify3.1 embedding: [16, 128], ours: [17, 128] — 只覆盖前16个
proj = nn.Linear(128, 128, bias=False)
with torch.no_grad():
    projected = proj(pt_emb)

print(f'V5.1: 连续链算术 | 1柱 → SparkTrigger → 自动增长')
print(f'd_model={cfg.d_model} batch={cfg.batch_size} lr={cfg.lr} max_epochs={cfg.max_epochs}')
train_data = gen_chain(20000, seed=42); val_data = gen_chain(2000, seed=999)
print(f'数据: train={len(train_data)} val={len(val_data)} '
      f'avg_len={sum(len(d) for d in train_data)/len(train_data):.0f}')

tl = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
    collate_fn=lambda b: collate(b))
vl = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False,
    collate_fn=lambda b: collate(b))

model = GerminalSpark(cfg)
with torch.no_grad(): model.embedding.weight[:16].copy_(projected)
# 注: embedding 目前可训练（未冻结），让模型自行适配新 token

params = list(model.parameters()) + list(proj.parameters())
opt = optim.AdamW(params, lr=cfg.lr)
sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)

best_ar = 0
for ep in range(cfg.max_epochs):
    model.train(); tl_sum, ta_sum, n = 0,0,0
    for x, t, m in tl:
        opt.zero_grad()
        logits, info = model(x, t, m)
        info['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl_sum += info['loss'].item(); ta_sum += compute_acc(logits, t, m); n += 1
    sch.step()

    model.eval(); va_sum, vn = 0,0
    with torch.no_grad():
        for x, t, m in vl:
            logits, info = model(x, t, m)
            va_sum += compute_acc(logits, t, m); vn += 1
    val_acc = va_sum / vn

    ar_acc = ar_eval(model, val_data, n=100)
    best_ar = max(best_ar, ar_acc)

    s = model.get_stats()
    util_str = ' '.join(f'{u:.1%}' for u in s['expert_utilization'])
    cols_str = f"{s['num_columns']}c/{s['mature_columns']}m"
    pe_act = model.column_pool.all_columns[0].get_pe_activity() if model.column_pool.all_columns else 0

    print(f'E{ep:2d} [{cols_str}] loss={tl_sum/n:.3f} ta={ta_sum/n:.2%} '
          f'va={val_acc:.2%} AR={ar_acc:.2%} PE={pe_act:.4f} util=[{util_str}]', flush=True)

    logs = model.check_growth(epoch=ep, val_acc=val_acc)
    for l in logs:
        print(f'  {l}', flush=True)

    for col in model.column_pool.all_columns:
        if hasattr(col, 'update_pe_prev'): col.update_pe_prev()

s = model.get_stats()
print(f'\nDone | best_AR={best_ar:.2%} | cols={s["num_columns"]}/{s["mature_columns"]}m | sparks={s["total_sparks"]}')
