"""Generate sample outputs from trained verify3.2 model."""
import torch
from train import *

device = torch.device('cpu')

# Load model
model = MixedPETransformer(VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                           num_layers=NUM_LAYERS, d_ff=D_FF, max_len=MAX_LEN, dropout=DROPOUT)
model.load_state_dict(torch.load('best.pth', map_location=device, weights_only=True))
model.eval()
print(f"Model loaded: 792,880 params\n")

# Test cases
tests = [
    "12+34=46", "99+1=100", "50-30=20", "7-9=-2",
    "0+0=0", "88+12=100", "45-99=-54", "5+5=10",
    "23+67=90", "100-50=50", "1-99=-98", "75+25=100",
    "33-33=0", "8-0=8", "99-0=99",
]

print(f"{'Problem':>12}  {'Expected':>10}  {'Generated':>10}  {'✓/✗'}")
print("-" * 50)

correct = 0
for test in tests:
    eq_pos = test.index('=')
    question = test[:eq_pos + 1]  # e.g. "12+34="
    expected = test[eq_pos + 1:]  # e.g. "46"

    # Encode question
    prefix = [TOKEN2ID['BOS']] + [TOKEN2ID[c] for c in question]  # BOS,1,2,+,3,4,=
    gen_ids = list(prefix)

    # Generate until EOS or max 5 tokens
    for _ in range(5):
        ids_tensor = torch.tensor([gen_ids], device=device)
        logits, _ = model(ids_tensor, None)
        next_token = logits[0, -1].argmax(dim=-1).item()
        gen_ids.append(next_token)
        if next_token == TOKEN2ID['EOS']:
            break

    # Decode generated answer
    gen_str = ''.join(ID2TOKEN[t] for t in gen_ids[len(prefix):])
    gen_str = gen_str.replace('EOS', '')

    ok = gen_str == expected
    if ok:
        correct += 1

    print(f"{test:>12}  {expected:>10}  {gen_str:>10}  {'✓' if ok else '✗'}")

print(f"\n{correct}/{len(tests)} correct")
