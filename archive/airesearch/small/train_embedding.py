"""
tiny — Token Embedding 预训练 (d_model=48)
与 verify3.1 相同方法，但维度压缩到 48
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VOCAB_SIZE = 16
D_MODEL = 48
N_EPOCHS = 2000
LR = 5e-3
SEED = 42

DIGIT_IDX = list(range(10))
OP_IDX = [10, 11]
SPECIAL_IDX = [12, 13, 14, 15]
TOKEN_NAMES = ['0','1','2','3','4','5','6','7','8','9','+','-','=','PAD','BOS','EOS']

torch.manual_seed(SEED)

class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB_SIZE, D_MODEL) * 0.1)
    def forward(self, x):
        return F.embedding(x, self.weight)

model = TokenEmbedding()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

def compute_loss(emb):
    emb_norm = F.normalize(emb, dim=1)
    digits = emb_norm[DIGIT_IDX]
    ops = emb_norm[OP_IDX]
    specials = emb_norm[SPECIAL_IDX]
    loss = 0.0

    # 1. digit cluster
    digit_sim = digits @ digits.T
    target_digit = torch.eye(10, device=emb.device) * 1.0 + (1 - torch.eye(10, device=emb.device)) * 0.35
    loss += F.mse_loss(digit_sim, target_digit) * 10.0

    # 2. op cluster
    op_sim = ops @ ops.T
    target_op = torch.eye(2, device=emb.device) * 1.0 + (1 - torch.eye(2, device=emb.device)) * 0.5
    loss += F.mse_loss(op_sim, target_op) * 5.0

    # 3. digit-op orthogonal
    loss += (digits @ ops.T).pow(2).mean() * 20.0

    # 4. special token separation
    loss += (digits @ specials.T).pow(2).mean() * 2.0
    loss += (ops @ specials.T).pow(2).mean() * 2.0

    # 5. numerical order
    margin_loss = 0.0
    n_pairs = 0
    for anchor in range(10):
        for pos in range(10):
            if pos == anchor: continue
            dist = abs(anchor - pos)
            for neg in range(10):
                if neg == anchor or abs(anchor - neg) <= dist: continue
                sim_pos = digits[anchor] @ digits[pos]
                sim_neg = digits[anchor] @ digits[neg]
                margin_loss += torch.relu(sim_neg - sim_pos + 0.05)
                n_pairs += 1
    if n_pairs > 0:
        loss += margin_loss / n_pairs * 2.0

    # 6. anti-collapse
    loss += torch.relu(0.05 - digits.std(dim=0).mean()) * 5.0
    return loss

print(f"Training {D_MODEL}-dim token embedding ({N_EPOCHS} epochs)...")
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    loss = compute_loss(model.weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 200 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            emb_norm = F.normalize(model.weight, dim=1)
            digits = emb_norm[DIGIT_IDX]
            ops = emb_norm[OP_IDX]
            d_d = (digits @ digits.T).mean().item()
            d_op = (digits @ ops.T).abs().mean().item()
            o_o = (ops @ ops.T)[0,1].item()
            adj = sum((digits[i] @ digits[i+1]).item() for i in range(9)) / 9
        print(f"{epoch:>6d} | loss={loss.item():.4f} | d-d={d_d:.3f} | d-op={d_op:.3f} | op-op={o_o:.3f} | adj={adj:.3f}")

# final analysis
with torch.no_grad():
    emb_norm = F.normalize(model.weight, dim=1)
    sim = emb_norm @ emb_norm.T
    digit_pairs = [sim[i,j].item() for i in range(10) for j in range(i+1,10)]
    print(f"\ndigit-digit similarity: {np.mean(digit_pairs):.4f}")
    d_op_pairs = [abs(sim[i,j].item()) for i in range(10) for j in [10,11]]
    print(f"digit-op |similarity|: {np.mean(d_op_pairs):.4f}")
    print(f"+ vs - similarity: {sim[10,11].item():.4f}")
    adj_ok = 0; total = 0
    for i in range(10):
        for d1 in range(1,5):
            for d2 in range(d1+1, min(6,10-i)):
                if i+d1<10 and i+d2<10:
                    if sim[i,i+d1] > sim[i,i+d2]: adj_ok += 1
                    total += 1
    print(f"order preservation: {adj_ok}/{total} = {adj_ok/total*100:.1f}%")

save_path = '/root/airesearch/small/embedding.pth'
torch.save(model.weight.data.clone(), save_path)
print(f"\nsaved: {save_path}")
