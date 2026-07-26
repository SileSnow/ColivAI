import torch, random
torch.manual_seed(42)
V = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'+':10,'-':11,'=':12,'PAD':13,'BOS':14,'EOS':15}
I = {v:k for k,v in V.items()}

from config import ARITHMETIC_CONFIG
from germinal_spark import GerminalSpark

cfg = ARITHMETIC_CONFIG
cfg.vocab_size = 16; cfg.initial_columns = 1; cfg.max_columns = 3
cfg.d_model = 128; cfg.d_ff = 512; cfg.n_heads = 4

pt = torch.load('/root/airesearch/verify3.1/embedding.pth', map_location='cpu', weights_only=False)
model = GerminalSpark(cfg)
with torch.no_grad(): model.embedding.weight[:16].copy_(pt)
model.embedding.weight.requires_grad = False
model.eval()

# 1. forward
x = torch.tensor([[14,8,6,10,1,0,12,7,6,15]])
t = torch.tensor([[8,6,10,1,0,12,7,6,15,15]])
m = torch.tensor([[0.,0,0,0,0,0,1,1,1,1]])
with torch.no_grad():
    logits, info = model(x, t, m)
print(f'forward: loss={info["task_loss"].item():.4f} cols={info["num_columns"]} logits={list(logits.shape)}')

# 2. =位置预测
eq_pos = 6
top3 = logits[0, eq_pos].topk(3)
preds = [(I[i.item()], f'{v:.1f}') for v,i in zip(top3.values, top3.indices)]
print(f'at =: top3={preds}')

# 3. AR
gen = [14,8,6,10,1,0,12]
for _ in range(6):
    xg = torch.tensor([gen], dtype=torch.long)
    with torch.no_grad():
        lg, _ = model(xg)
    nxt = lg[0,-1].argmax().item()
    gen.append(nxt)
    if nxt == 15: break
output = ''.join(I[t] for t in gen[7:])
print(f'AR: 86+10= -> "{output}"')

# 4. 柱状态
for i, col in enumerate(model.column_pool.all_columns):
    print(f'col[{i}]: layers={col.num_layers} mature={col.is_mature}')

# 5. 训练不报错
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
opt.zero_grad()
logits, info = model(x, t, m)
info['loss'].backward()
print(f'train: loss={info["loss"].item():.4f} grad OK')
print('ALL PASSED')
