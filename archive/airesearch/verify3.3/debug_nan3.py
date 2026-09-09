"""Debug NaN - test with batch_size=16."""
import torch, math, sys
sys.path.insert(0, '/root/airesearch/verify3.3')
from train import *

set_seed(42)
device = torch.device('cpu')

chains = [generate_chain() for _ in range(20)]
encoded = [encode(c) for c in chains]
ids_t, mask_t = collate_batch(encoded[:16])
print(f'Batch: {ids_t.shape}, max_len={ids_t.shape[1]}')

# Check lengths
lens = [len(e) for e in encoded[:16]]
print(f'Lengths: min={min(lens)} max={max(lens)}')

model = MixedPETransformer(VOCAB_SIZE, d_model=128, nhead=4, num_layers=4, d_ff=512, max_len=128, dropout=0.0)
model.init_weights()
model.train()

# Check active loss tokens
shift_mask = mask_t[:, :-1]
n_active = shift_mask.bool().sum().item()
print(f'Active loss tokens: {n_active}')

_, loss = model(ids_t, mask_t)
print(f'Forward loss: {loss.item():.4f}, NaN: {torch.isnan(loss).item()}')

if not torch.isnan(loss):
    loss.backward()
    nan_count = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    print(f'NaN grads: {nan_count}, Max grad: {max_grad:.2e}')
else:
    print('Loss is NaN!')
