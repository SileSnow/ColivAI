"""基线：标准 Transformer + 连续链算术"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from torch.optim import AdamW

V = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'+':10,'-':11,'=':12,'\n':13,'PAD':14,'BOS':15,'EOS':16}
I = {v:k for k,v in V.items()}

def enc(t): return [V.get(c,14) for c in t]
def gen(n,s,min_s=2,max_s=8):
    random.seed(s); d=[]
    for _ in range(n):
        steps=random.randint(min_s,max_s); a=random.randint(1,99)
        lines=[]
        for _ in range(steps):
            op=random.choice(['+','-']); b=random.randint(1,99)
            if op=='-': a,b=max(a,b),min(a,b); r=a-b
            else: r=a+b
            lines.append(f'{a}{op}{b}={r}'); a=r
        tokens=[15]+enc('\n'.join(lines)+f'\n={r}')+[16]
        d.append(tokens)
    return d

def collate(batch, max_len=128, pad=14):
    batch=[b[:max_len] for b in batch]; L=max(len(b) for b in batch)
    x=torch.full((len(batch),L),pad,dtype=torch.long)
    m=torch.zeros(len(batch),L)
    for i,seq in enumerate(batch):
        x[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        for j in range(len(seq)-1,-1,-1):
            if seq[j]==12: m[i,j:len(seq)]=1.0; break
    t=torch.full_like(x,pad); t[:,:-1]=x[:,1:].clone(); t[:,-1]=16
    return x,t,m

class M(nn.Module):
    def __init__(self, vocab=17, d=128, nh=4, nl=4, dff=512):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        pe=torch.zeros(128,d); pos=torch.arange(128).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe)
        el=nn.TransformerEncoderLayer(d,nh,dff,batch_first=True,dropout=0.0,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.head=nn.Linear(d,vocab,bias=False)
    def forward(self,x):
        B,T=x.shape; h=self.emb(x)+self.pe[:T]
        return self.head(self.enc(h,mask=nn.Transformer.generate_square_subsequent_mask(T,device=x.device),is_causal=True))

torch.manual_seed(42); random.seed(42)
td=gen(20000,42); vd=gen(2000,999)
tl=torch.utils.data.DataLoader(td,16,shuffle=True,collate_fn=lambda b:collate(b))
vl=torch.utils.data.DataLoader(vd,16,shuffle=False,collate_fn=lambda b:collate(b))

m=M()
print(f'参数: {sum(p.numel() for p in m.parameters()):,}')
opt=AdamW(m.parameters(),lr=3e-4)

best_ar=0
for ep in range(30):
    m.train(); tl_sum,ta_sum,n=0,0,0
    for x,t,mk in tl:
        opt.zero_grad(); logits=m(x); act=mk.view(-1)==1
        loss=F.cross_entropy(logits.view(-1,17)[act],t.view(-1)[act]) if act.any() else torch.tensor(0.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        tl_sum+=loss.item(); ta_sum+=(logits.view(-1,17)[act].argmax(-1)==t.view(-1)[act]).float().mean().item(); n+=1
    
    m.eval(); va_sum,vn=0,0
    with torch.no_grad():
        for x,t,mk in vl:
            logits=m(x); act=mk.view(-1)==1
            va_sum+=(logits.view(-1,17)[act].argmax(-1)==t.view(-1)[act]).float().mean().item(); vn+=1
    
    # AR
    correct=0
    for tokens in vd[:100]:
        eqs=[j for j,t in enumerate(tokens) if t==12]
        if not eqs: continue
        prefix=tokens[:eqs[-1]+1]
        target=''.join(I[t] for t in tokens[eqs[-1]+1:] if t not in[16,13])
        g=list(prefix)
        with torch.no_grad():
            for _ in range(10):
                xg=torch.tensor([g[-128:]],dtype=torch.long)
                nxt=m(xg)[0,-1].argmax().item(); g.append(nxt)
                if nxt==16: break
        if ''.join(I[t] for t in g[len(prefix):] if t not in[16,13])==target: correct+=1
    ar=correct/100; best_ar=max(best_ar,ar)
    print(f'E{ep:2d} loss={tl_sum/n:.3f} ta={ta_sum/n:.2%} va={va_sum/vn:.2%} AR={ar:.2%}',flush=True)

print(f'\nbest_AR={best_ar:.2%}')
