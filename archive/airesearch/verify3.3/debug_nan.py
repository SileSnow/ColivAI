"""Debug NaN in chain training."""
import torch, math, sys
sys.path.insert(0, '/root/airesearch/verify3.3')
from train import *

set_seed(42)
device = torch.device('cpu')

# Generate a few chains
chains = [generate_chain() for _ in range(4)]
encoded = [encode(c) for c in chains]

# Try with batch of 2
ids_t, mask_t = collate_batch(encoded[:2])
print(f'Batch shape: {ids_t.shape}, max_len={ids_t.shape[1]}')

model = MixedPETransformer(VOCAB_SIZE, d_model=128, nhead=4, num_layers=4, d_ff=512, max_len=128, dropout=0.0)
model.init_weights()
model.eval()

with torch.no_grad():
    logits, loss = model(ids_t, mask_t)
    print(f'Has NaN: {torch.isnan(logits).any().item()}')
    print(f'Has Inf: {torch.isinf(logits).any().item()}')
    if loss is not None:
        print(f'Loss: {loss.item():.4f}, isNaN: {torch.isnan(loss).item()}')

# Try with training mode (which matters for layernorm stats?)
model.train()
with torch.no_grad():
    logits2, loss2 = model(ids_t, mask_t)
    print(f'Train mode - Has NaN: {torch.isnan(logits2).any().item()}, Loss NaN: {torch.isnan(loss2).item() if loss2 is not None else False}')
