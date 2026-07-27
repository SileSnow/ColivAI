import torch, random
torch.manual_seed(42)
from train_v51 import gen_chain, collate, V, I

data = gen_chain(100, min_s=2, max_s=4)
x,t,m = collate(data[:4])
print(f'data: {len(data)} x={list(x.shape)}')

sample = data[0]
eq_pos = max(j for j,tok in enumerate(sample) if tok==V['='])
print(f'sample len={len(sample)} last_eq={eq_pos}')

from config import ARITHMETIC_CONFIG
from germinal_spark import GerminalSpark
cfg = ARITHMETIC_CONFIG
cfg.vocab_size = 17; cfg.initial_columns = 1; cfg.max_columns = 4
cfg.d_model = 128; cfg.d_ff = 512; cfg.n_heads = 4
model = GerminalSpark(cfg)
model.eval()
with torch.no_grad():
    logits, info = model(x, t, m)
print(f'fwd loss={info["task_loss"].item():.3f} cols={info["num_columns"]}')
col = model.column_pool.all_columns[0]
print(f'col layers={col.num_layers} mature={col.is_mature}')
print('SMOKE PASSED')
