"""Debug NaN in backward pass."""
import torch, math, sys
sys.path.insert(0, '/root/airesearch/verify3.3')
from train import *

set_seed(42)
device = torch.device('cpu')

chains = [generate_chain() for _ in range(4)]
encoded = [encode(c) for c in chains]
ids_t, mask_t = collate_batch(encoded[:2])
print(f'Batch: {ids_t.shape}')

model = MixedPETransformer(VOCAB_SIZE, d_model=128, nhead=4, num_layers=4, d_ff=512, max_len=128, dropout=0.0)
model.init_weights()
model.train()

_, loss = model(ids_t, mask_t)
print(f'Forward loss: {loss.item():.4f}')

loss.backward()

# Check gradients
nan_params = []
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        nan_params.append(name)
        print(f'  NaN grad: {name}, shape={param.grad.shape}, norm={param.grad.norm():.2e}')

if nan_params:
    print(f'\n{len(nan_params)} params with NaN gradients!')
else:
    print('All gradients OK')

# Check max gradient magnitude
max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
print(f'Max grad magnitude: {max_grad:.2e}')
