import torch, torch.nn as nn, torch.nn.functional as F, math, random

VOCAB = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
         '+':10,'-':11,'=':12,'PAD':13,'BOS':14,'EOS':15}
ID2CHAR = {v:k for k,v in VOCAB.items()}

def encode(text):
    return [VOCAB.get(c, VOCAB['PAD']) for c in text]

def gen_data(n, seed):
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
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 128)
        pe = torch.zeros(32, 128)
        pos = torch.arange(32, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0,128,2,dtype=torch.float32)*(-math.log(10000.0)/128))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe)
        dl = nn.TransformerDecoderLayer(d_model=128,nhead=4,dim_feedforward=512,batch_first=True,dropout=0.0,activation='gelu')
        self.decoder = nn.TransformerDecoder(dl, num_layers=4)
        self.lm_head = nn.Linear(128, 16, bias=False)
    def forward(self, x):
        B,T=x.shape; device=x.device
        h = self.embedding(x) + self.pe[:T,:]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        h = self.decoder(h, h, tgt_mask=mask)
        return self.lm_head(h)

torch.manual_seed(42); random.seed(42)
pt_emb = torch.load('/root/airesearch/verify3.1/embedding.pth', map_location='cpu', weights_only=False)

train_data=gen_data(20000,42)
tl=torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,collate_fn=lambda b:collate(b))

model=BaselineModel()
with torch.no_grad(): model.embedding.weight[:16].copy_(pt_emb)
model.embedding.weight.requires_grad=False

opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5)

print('快速训练...')
for ep in range(5):
    model.train()
    for x,t,m in tl:
        opt.zero_grad()
        logits=model(x)
        active=m.view(-1)==1
        if active.any():
            loss=F.cross_entropy(logits.view(-1,16)[active],t.view(-1)[active])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    sch.step()
    print(f'  E{ep} loss={loss.item():.4f}', flush=True)

model.eval()
print('\n=== 诊断: teacher-forcing 各位置预测 ===')
for expr in ['86-10=', '42-17=']:
    tokens = [VOCAB['BOS']] + encode(expr)
    x = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    print(f'\n{expr}:')
    for i, tok in enumerate(tokens):
        pred_id = logits[0,i].argmax().item()
        tgt = tokens[i+1] if i+1 < len(tokens) else VOCAB['EOS']
        ok = '✅' if pred_id == tgt else '❌'
        print(f'  pos{i} see={ID2CHAR[tok]:>3} pred={ID2CHAR[pred_id]:>3} tgt={ID2CHAR[tgt]:>3} {ok}')

print('\n=== 诊断: 自回归 ===')
for expr in ['86-10=', '42-17=']:
    tokens = [VOCAB['BOS']] + encode(expr)
    gen = list(tokens)
    with torch.no_grad():
        for step in range(10):
            x = torch.tensor([gen[-32:]], dtype=torch.long)
            logits = model(x)
            nxt = logits[0,-1].argmax().item()
            gen.append(nxt)
            if nxt == VOCAB['EOS']: break
    raw = [ID2CHAR[t] for t in gen]
    pred = ''.join(ID2CHAR[t] for t in gen[len(tokens):] if t != VOCAB['EOS'])
    print(f'  {expr} → gen={raw} pred="{pred}"')
