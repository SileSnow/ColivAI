"""对照实验：标准 Transformer + 因果掩码 + 解耦合embedding"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

VOCAB = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
         '+':10,'-':11,'=':12,'PAD':13,'BOS':14,'EOS':15}
ID2CHAR = {v:k for k,v in VOCAB.items()}

def encode(text):
    return [VOCAB.get(c, VOCAB['PAD']) for c in text]

def gen_data(n=20000, seed=42):
    random.seed(seed); data=[]
    for _ in range(n):
        a,b=random.randint(0,99),random.randint(0,99)
        op=random.choice(['+','-'])
        if op=='-': a,b=max(a,b),min(a,b)
        c=a+b if op=='+' else a-b
        data.append([VOCAB['BOS']]+encode(f'{a}{op}{b}={c}')+[VOCAB['EOS']])
    return data

def collate(batch, max_len=32, pad=13):
    batch=[b[:max_len] for b in batch]; L=max(len(b) for b in batch)
    x=torch.full((len(batch),L),pad,dtype=torch.long)
    mask=torch.zeros(len(batch),L)
    for i,seq in enumerate(batch):
        x[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        for j in range(len(seq)-1,-1,-1):
            if seq[j]==VOCAB['=']: mask[i,j:]=1.0; break
    targets=torch.full_like(x,pad)
    targets[:,:-1]=x[:,1:].clone(); targets[:,-1]=VOCAB['EOS']
    return x,targets,mask

class BaselineModel(nn.Module):
    def __init__(self, vocab=16, d_model=128, nhead=4, nlayers=4, d_ff=512, max_len=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            batch_first=True, dropout=0.0, activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
    def forward(self, x):
        B,T=x.shape; device=x.device
        h = self.embedding(x) + self.pe[:T,:]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        h = self.decoder(h, h, tgt_mask=mask)
        return self.lm_head(h)

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
            logits=model(x); nxt=logits[0,-1].argmax().item()
            gen.append(nxt)
            if nxt==VOCAB['EOS']: break
        pred=''.join(ID2CHAR[t] for t in gen[len(prefix):] if t!=VOCAB['EOS'])
        if pred==target: correct+=1
    return correct/n

def compute_acc(logits, targets, mask):
    active=mask.view(-1)==1
    if not active.any(): return 0.0
    return (logits.view(-1,logits.shape[-1])[active].argmax(-1)==
            targets.view(-1)[active]).float().mean().item()

torch.manual_seed(42); random.seed(42)

pt_emb = torch.load('/root/airesearch/verify3.1/embedding.pth', map_location='cpu', weights_only=False)
print(f'预训练 embedding: {pt_emb.shape}')

train_data=gen_data(20000,42); val_data=gen_data(2000,999)
tl=torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,collate_fn=lambda b:collate(b))
vl=torch.utils.data.DataLoader(val_data,batch_size=32,shuffle=False,collate_fn=lambda b:collate(b))

model=BaselineModel(vocab=16,d_model=128,nhead=4,nlayers=4,d_ff=512)
with torch.no_grad(): model.embedding.weight[:16].copy_(pt_emb)
model.embedding.weight.requires_grad=False
print(f'参数: {sum(p.numel() for p in model.parameters()):,} (embedding冻结)')

opt=AdamW(model.parameters(),lr=1e-3)
sch=CosineAnnealingLR(opt,T_max=30)

best_ar=0
for ep in range(30):
    model.train(); tl_sum,ta_sum,n=0,0,0
    for x,t,m in tl:
        opt.zero_grad()
        logits=model(x)
        active=m.view(-1)==1
        if active.any():
            loss=F.cross_entropy(logits.view(-1,16)[active],t.view(-1)[active])
        else: loss=torch.tensor(0.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        tl_sum+=loss.item(); ta_sum+=compute_acc(logits,t,m); n+=1
    sch.step()
    model.eval(); va_sum,vn=0,0
    with torch.no_grad():
        for x,t,m in vl: logits=model(x); va_sum+=compute_acc(logits,t,m); vn+=1
    ar_acc=ar_eval(model,val_data,200)
    best_ar=max(best_ar,ar_acc)
    print(f'E{ep:2d} loss={tl_sum/n:.3f} ta={ta_sum/n:.2%} va={va_sum/vn:.2%} AR={ar_acc:.2%}',flush=True)

print(f'\nDone | best_AR={best_ar:.2%}')
