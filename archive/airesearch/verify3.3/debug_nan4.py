"""Debug NaN - run exact training loop with anomaly detection."""
import torch, math, sys
sys.path.insert(0, '/root/airesearch/verify3.3')
from train import *

set_seed(42)
device = torch.device('cpu')

print("Generating data...")
train_raw = generate_dataset(200)
val_raw = generate_dataset(50)
train_encoded = [encode(s) for s in train_raw]
val_encoded = [encode(s) for s in val_raw]

model = MixedPETransformer(VOCAB_SIZE, d_model=128, nhead=4, num_layers=4, d_ff=512, max_len=128, dropout=0.0)
model.init_weights()
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

torch.autograd.set_detect_anomaly(True)

print("Running batches...")
for i in range(0, len(train_encoded), 16):
    batch = train_encoded[i:i+16]
    ids_t, mask_t = collate_batch(batch)
    
    try:
        _, loss = model(ids_t, mask_t)
        print(f"  Batch {i//16}: loss={loss.item():.4f}, shape={ids_t.shape}, active={mask_t[:,:-1].bool().sum().item()}")
        
        if torch.isnan(loss):
            print(f"  ⚠ NaN detected!")
            # Print first chain
            print(f"  First chain: {repr(train_raw[i][:80])}")
            break
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    except RuntimeError as e:
        print(f"  ⚠ Error at batch {i//16}: {e}")
        break

print("Done")
