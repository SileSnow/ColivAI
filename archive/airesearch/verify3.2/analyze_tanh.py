"""Analyze tanh/sech² parameters from trained verify3.2 model."""
import torch
import math
from train import *

device = torch.device('cpu')
model = MixedPETransformer(VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                           num_layers=NUM_LAYERS, d_ff=D_FF, max_len=MAX_LEN, dropout=DROPOUT)
model.load_state_dict(torch.load('best.pth', map_location=device, weights_only=True))
model.eval()

print("=" * 70)
print("verify3.2 — tanh/sech² relative bias analysis")
print("=" * 70)

for layer_idx in range(4):
    layer = model.layers[layer_idx]
    attn = layer.attn
    print(f"\n{'─'*60}")
    print(f"Layer {layer_idx}")
    print(f"{'─'*60}")
    print(f"{'Head':>6} {'w':>10} {'v':>10} {'τ':>10} {'shape':>20}")
    print(f"{'─'*60}")

    for h in range(4):
        w = attn.w_param[h].item()
        v = attn.v_param[h].item()
        tau = attn.log_tau.exp()[h].item()

        # Describe the shape
        if abs(w) < 0.01:
            shape = "dormant (w≈0)"
        elif abs(v) < 0.01:
            shape = "flat (v≈0, weak distance sensitivity)"
        elif tau > 5:
            shape = "ultra-sharp (τ>>1, d=0 only)"
        elif tau > 2:
            shape = "sharp nearby focus"
        elif tau > 0.5:
            shape = "moderate range"
        else:
            shape = "long-range (τ<1)"

        print(f"  H{h:>3}  {w:>+10.4f}  {v:>+10.4f}  {tau:>10.4f}  {shape:>20}")

# Show bias curves for a few distances
print(f"\n{'='*70}")
print("Bias values for distances d=0..10 (per head, per layer)")
print(f"{'='*70}")

for layer_idx in range(4):
    attn = model.layers[layer_idx].attn
    print(f"\nLayer {layer_idx}:")
    header = f"{'d':>4}"
    for h in range(4):
        header += f"  {'H'+str(h):>12}"
    print(header)
    print("-" * (4 + 14 * 4))

    for d in range(11):
        row = f"{d:>4}"
        for h in range(4):
            w = attn.w_param[h].item()
            v = attn.v_param[h].item()
            tau = attn.log_tau.exp()[h].item()
            bias = w * math.tanh(v * d + 1e-8) + 1.0 / math.cosh(tau * d) ** 2
            row += f"  {bias:>+12.4f}"
        print(row)
