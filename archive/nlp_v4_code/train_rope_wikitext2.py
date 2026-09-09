import os, math, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import sentencepiece as spm
from model_rope import TransformerLM_RoPE

DATA_DIR   = '/root/nlp_v4/data'
LOG_DIR    = '/root/nlp_v4/logs'
MODEL_DIR  = '/root/nlp_v4/models'
BPE_PATH   = DATA_DIR + '/spm_bpe_2000.model'
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

VOCAB = 2000; PAD,BOS,EOS = 0,1,2; SEQ=512; BS=32; EPOCHS=8; LR=3e-4; LR_MIN=1e-5; WARMUP=300

class DS(torch.utils.data.Dataset):
    def __init__(s,c): s.x=c[:,:-1]; s.y=c[:,1:]
    def __len__(s): return len(s.x)
    def __getitem__(s,i): return s.x[i],s.y[i]

train = torch.load(DATA_DIR+'/train.pt',weights_only=True)
valid = torch.load(DATA_DIR+'/valid.pt',weights_only=True)
tl = DataLoader(DS(train),BS,shuffle=True)
vl = DataLoader(DS(valid),BS)
steps = len(tl)*EPOCHS
print(f'[Data] train={train.shape} valid={valid.shape} total_steps={steps}')

m = TransformerLM_RoPE(VOCAB, max_len=600)
print(f'[Model] {m.count_parameters():,} params')

opt = torch.optim.AdamW(m.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.01)
sp = spm.SentencePieceProcessor(); sp.load(BPE_PATH)

best=float('inf'); gs=0
for ep in range(1,EPOCHS+1):
    m.train(); t0=time.time()
    for x,y in tl:
        if gs<WARMUP: lr=LR*gs/WARMUP
        else: prog=(gs-WARMUP)/max(1,steps-WARMUP); lr=LR_MIN+0.5*(LR-LR_MIN)*(1+math.cos(math.pi*prog))
        for pg in opt.param_groups: pg['lr']=lr
        opt.zero_grad()
        loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=PAD)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        gs+=1
        if gs%100==0 or gs==1:
            print(f'[Step {gs:5d}/{steps}] ppl={math.exp(loss.item()):.1f} lr={lr:.2e} {time.time()-t0:.0f}s')
        if gs%500==0:
            m.eval(); vl_sum=vl_n=0
            with torch.no_grad():
                for vx,vy in vl: 
                    vl_l=F.cross_entropy(m(vx).reshape(-1,VOCAB),vy.reshape(-1),ignore_index=PAD)
                    vl_sum+=vl_l.item()*vx.numel(); vl_n+=vx.numel()
            vp=math.exp(vl_sum/vl_n); print(f'  [Eval] Val PPL={vp:.2f}')
            if vp<best: best=vp; torch.save(m.state_dict(),MODEL_DIR+'/best_rope_wt2.pt'); print(f'  [Save]')
            m.train()
    print(f'[Epoch {ep}] {time.time()-t0:.0f}s')
print(f'[Done] Best Val PPL: {best:.2f}')
