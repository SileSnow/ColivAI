import torch, torch.nn as nn, torch.nn.functional as F, math, random
V = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'+':10,'-':11,'=':12,'PAD':13,'BOS':14,'EOS':15}
I = {v:k for k,v in V.items()}
def enc(t): return [V.get(c,13) for c in t]
def gen(n,s):
    random.seed(s); d=[]
    for _ in range(n):
        a,b=random.randint(0,99),random.randint(0,99); op=random.choice(['+','-'])
        if op=='-': a,b=max(a,b),min(a,b); c=a-b
        else: c=a+b
        d.append([14]+enc(f'{a}{op}{b}={c}')+[15])
    return d
def coll(batch, pad=13):
    batch=[b[:32] for b in batch]; L=max(len(b) for b in batch)
    x=torch.full((len(batch),L),pad,dtype=torch.long)
    m=torch.zeros(len(batch),L)
    for i,seq in enumerate(batch):
        x[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        for j in range(len(seq)-1,-1,-1):
            if seq[j]==12: m[i,j:]=1.0; break
    t=torch.full_like(x,pad); t[:,:-1]=x[:,1:].clone(); t[:,-1]=15
    return x,t,m

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(16,128)
        pe=torch.zeros(32,128); pos=torch.arange(32).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,128,2).float()*(-math.log(10000)/128))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe)
        el=nn.TransformerEncoderLayer(d_model=128,nhead=4,dim_feedforward=512,batch_first=True,dropout=0.0,activation='gelu')
        self.enc=nn.TransformerEncoder(el,4); self.head=nn.Linear(128,16,bias=False)
    def forward(self,x):
        B,T=x.shape; device=x.device; h=self.emb(x)+self.pe[:T]
        mask=nn.Transformer.generate_square_subsequent_mask(T,device=device)
        return self.head(self.enc(h,mask=mask,is_causal=True))

torch.manual_seed(42); random.seed(42)
pt=torch.load('/root/airesearch/verify3.1/embedding.pth',map_location='cpu',weights_only=False)
td=gen(20000,42); tl=torch.utils.data.DataLoader(td,32,shuffle=True,collate_fn=lambda b:coll(b))
m=M()
with torch.no_grad(): m.emb.weight[:16].copy_(pt)
m.emb.weight.requires_grad=False
opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
print('训练10epoch...')
for ep in range(10):
    m.train()
    for x,t,mk in tl:
        opt.zero_grad(); logits=m(x); act=mk.view(-1)==1
        if act.any(): loss=F.cross_entropy(logits.view(-1,16)[act],t.view(-1)[act])
        else: loss=torch.tensor(0.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    print(f'E{ep} loss={loss.item():.4f}',flush=True)
torch.save(m.state_dict(),'/root/germinal_spark/baseline_trained.pt')

# 诊断
m.eval()
print('\n=== TF各位置预测 ===')
for expr in ['86-10=','42-17=']:
    tokens=[14]+enc(expr); x=torch.tensor([tokens])
    with torch.no_grad(): logits=m(x)
    for i,tok in enumerate(tokens):
        pid=logits[0,i].argmax().item()
        tgt=tokens[i+1] if i+1<len(tokens) else 15
        ok = 'Y' if pid==tgt else 'N'
        print(f'  pos{i} see={I[tok]} pred={I[pid]} tgt={I[tgt]} {ok}')

print('\n=== 自回归生成 ===')
for expr in ['86-10=','42-17=','99+88=','7-3=']:
    tokens=[14]+enc(expr); gen=list(tokens)
    with torch.no_grad():
        for _ in range(10):
            x=torch.tensor([gen[-32:]],dtype=torch.long)
            nxt=m(x)[0,-1].argmax().item(); gen.append(nxt)
            if nxt==15: break
    pred=''.join(I[t] for t in gen[len(tokens):] if t!=15)
    print(f'  {expr} → \"{pred}\"')
