"""Phase 0: 单柱 + 因果掩码 + 解耦合embedding → 验证自回归"""
import torch, torch.nn as nn, torch.optim as optim, random
from torch.optim.lr_scheduler import CosineAnnealingLR

VOCAB = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
         '+':10,'-':11,'=':12,'PAD':13,'BOS':14,'EOS':15}
ID2CHAR = {v:k for k,v in VOCAB.items()}

def encode(text):
    return [VOCAB.get(c, VOCAB['PAD']) for c in text]

def generate_arithmetic(num_samples=20000, seed=42):
    random.seed(seed); data = []
    for _ in range(num_samples):
        a,b=random.randint(0,99),random.randint(0,99)
        op=random.choice(['+','-'])
        if op=='-': a,b=max(a,b),min(a,b)
        c=a+b if op=='+' else a-b
        tokens=[VOCAB['BOS']]+encode(f'{a}{op}{b}={c}')+[VOCAB['EOS']]
        data.append(tokens)
    return data

def collate_batch(batch, max_len=32, pad_id=13):
    batch=[b[:max_len] for b in batch]; max_l=max(len(b) for b in batch)
    x=torch.full((len(batch),max_l),pad_id,dtype=torch.long)
    mask=torch.zeros(len(batch),max_l)
    for i,seq in enumerate(batch):
        x[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        for j in range(len(seq)-1,-1,-1):
            if seq[j]==VOCAB['=']: mask[i,j:]=1.0; break
    # Shift targets: position i predicts token i+1 (standard autoregressive)
    targets = torch.full_like(x, pad_id)
    targets[:, :-1] = x[:, 1:].clone()
    targets[:, -1] = VOCAB['EOS']
    return x, targets, mask

def compute_acc(logits,targets,mask):
    active=mask.view(-1)==1
    if not active.any(): return 0.0
    return (logits.view(-1,logits.shape[-1])[active].argmax(-1)==
            targets.view(-1)[active]).float().mean().item()

@torch.no_grad()
def ar_eval(model, data, n=200):
    model.eval(); correct=0
    for tokens in data[:n]:
        eqs=[j for j,t in enumerate(tokens) if t==VOCAB['=']]
        if len(eqs)<1: continue
        prefix=tokens[:eqs[-1]+1]
        target=''.join(ID2CHAR[t] for t in tokens[eqs[-1]+1:] if t!=VOCAB['EOS'])
        gen=list(prefix)
        for _ in range(8):
            x=torch.tensor([gen[-32:]],dtype=torch.long)
            logits,_=model(x)
            nxt=logits[0,-1].argmax().item()
            gen.append(nxt)
            if nxt==VOCAB['EOS']: break
        pred=''.join(ID2CHAR[t] for t in gen[len(prefix):] if t!=VOCAB['EOS'])
        if pred==target: correct+=1
    return correct/n

from config import ARITHMETIC_CONFIG
from germinal_spark import GerminalSpark

torch.manual_seed(42); random.seed(42)
cfg=ARITHMETIC_CONFIG
cfg.vocab_size=16; cfg.initial_columns=1; cfg.max_columns=3; cfg.max_epochs=30
cfg.batch_size=32; cfg.lr=1e-3; cfg.consolidation_interval=999
cfg.d_model=128; cfg.d_ff=512; cfg.n_heads=4; cfg.d_head=32

# === 加载预训练 embedding（d_model=128，直接匹配）===
pt_emb = torch.load('/root/airesearch/verify3.1/embedding.pth',
                     map_location='cpu', weights_only=False)
print(f'预训练 embedding: {pt_emb.shape} (16x128) → 直接匹配 d_model=128')

print(f'Phase 0: 单柱 + 因果掩码 + 解耦合embedding')
train_data=generate_arithmetic(20000,seed=42)
val_data=generate_arithmetic(2000,seed=999)
tl=torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,
    collate_fn=lambda b:collate_batch(b))
vl=torch.utils.data.DataLoader(val_data,batch_size=32,shuffle=False,
    collate_fn=lambda b:collate_batch(b))

model=GerminalSpark(cfg)
with torch.no_grad(): model.embedding.weight[:16].copy_(pt_emb)
model.embedding.weight.requires_grad = False  # 冻结！

opt=optim.AdamW(model.parameters(),lr=cfg.lr)
sch=CosineAnnealingLR(opt,T_max=cfg.max_epochs)

best_ar=0
for ep in range(cfg.max_epochs):
    model.train(); tl_sum,ta_sum,n=0,0,0
    for x,t,m in tl:
        opt.zero_grad()
        logits,info=model(x,t,m)
        info['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        tl_sum+=info['loss'].item(); ta_sum+=compute_acc(logits,t,m); n+=1
    sch.step()
    
    model.eval(); va_sum,vn=0,0
    with torch.no_grad():
        for x,t,m in vl:
            logits,info=model(x,t,m); va_sum+=compute_acc(logits,t,m); vn+=1
    
    ar_acc=ar_eval(model,val_data,200)
    best_ar=max(best_ar,ar_acc)
    s=model.get_stats()
    print(f'E{ep:2d} [{s["num_columns"]}c] loss={tl_sum/n:.3f} '
          f'ta={ta_sum/n:.2%} va={va_sum/vn:.2%} AR={ar_acc:.2%}',flush=True)

print(f'\nDone | best_AR={best_ar:.2%}')
