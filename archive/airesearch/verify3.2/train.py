"""
verify3.2: Mixed Positional Encoding (sin/cos + tanh/sech²) — 1-step ± arithmetic
================================================================
- sin/cos 绝对位置编码（冻结，加在 token embedding 上）
- tanh/sech² 相对位置偏置（可学习，每头独立，加在注意力 logit 上）
- 任务：1-step ± 算术，0-99 范围
- 目标：验证混合编码可行，与 baseline（纯 sin/cos）直接对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# CONFIG
# ============================================================
VOCAB = ['0','1','2','3','4','5','6','7','8','9','+','-','=','PAD','BOS','EOS']
TOKEN2ID = {c: i for i, c in enumerate(VOCAB)}
ID2TOKEN = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)  # 16

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.0
MAX_LEN = 32

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 200
EARLY_STOP_ACC = 0.99

TRAIN_SIZE = 20000
VAL_SIZE = 2000

SEED = 42

# ============================================================
# UTILS
# ============================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# DATA
# ============================================================
def generate_sample():
    a = random.randint(0, 99)
    b = random.randint(0, 99)
    op = random.choice(['+', '-'])
    c = a + b if op == '+' else a - b
    return f"{a}{op}{b}={c}"

def generate_dataset(size):
    return [generate_sample() for _ in range(size)]

def encode(text):
    """text: e.g. '12+34=46' → [BOS, 1, 2, +, 3, 4, =, 4, 6, EOS]"""
    ids = [TOKEN2ID['BOS']]
    for ch in text:
        ids.append(TOKEN2ID[ch])
    ids.append(TOKEN2ID['EOS'])
    return ids

def make_loss_mask(ids):
    """Mark PREDICTION positions whose target is an answer token.
    Prediction at position i targets token i+1.
    Answer tokens are at eq_pos..T-1, so prediction positions are eq_pos..T-2."""
    eq_pos = ids.index(TOKEN2ID['='])
    mask = [0] * len(ids)
    for i in range(eq_pos, len(ids) - 1):
        mask[i] = 1
    return mask

def collate_batch(samples):
    """Pad batch of encoded samples, return (token_ids, loss_mask) tensors."""
    max_len = max(len(s) for s in samples)
    padded_ids = []
    padded_masks = []
    for s in samples:
        pad_len = max_len - len(s)
        padded_ids.append(s + [TOKEN2ID['PAD']] * pad_len)
        m = make_loss_mask(s)
        padded_masks.append(m + [0] * pad_len)
    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(padded_masks, dtype=torch.float32),
    )

# ============================================================
# MODEL
# ============================================================
class MixedAttention(nn.Module):
    """Multi-head attention with tanh/sech² relative position bias."""

    def __init__(self, d_model, nhead, max_len=MAX_LEN):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-head learnable relative bias: w·tanh(v·d) + sech²(τ·d)
        self.w_param = nn.Parameter(torch.zeros(nhead))
        self.v_param = nn.Parameter(torch.zeros(nhead))
        self.log_tau = nn.Parameter(torch.zeros(nhead))

        self.max_len = max_len

    def forward(self, x, causal_mask):
        B, T, D = x.shape
        H = self.nhead
        dk = self.head_dim

        q = self.q_proj(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk
        k = self.k_proj(x).view(B, T, H, dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dk).transpose(1, 2)

        scale = math.sqrt(dk)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale  # B,H,T,T

        # --- Relative position bias (tanh + sech²) ---
        device = x.device
        pos = torch.arange(T, device=device, dtype=torch.float32)
        distances = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)  # T,T

        w = self.w_param.view(H, 1, 1)       # H,1,1
        v_p = self.v_param.view(H, 1, 1)      # H,1,1
        tau = self.log_tau.exp().view(H, 1, 1)  # H,1,1

        tanh_bias = w * torch.tanh(v_p * distances + 1e-8)          # H,T,T
        sech2_bias = 1.0 / torch.cosh(tau * distances) ** 2         # H,T,T
        rel_bias = tanh_bias + sech2_bias                            # H,T,T

        attn_logits = attn_logits + rel_bias.unsqueeze(0)            # B,H,T,T

        # Causal mask
        attn_logits = attn_logits.masked_fill(~causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights, v)  # B,H,T,dk
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MixedAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        x = x + self.dropout(self.attn(self.norm1(x), causal_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MixedPETransformer(nn.Module):
    """
    Transformer with mixed positional encoding:
    - sin/cos absolute PE added to token embeddings (frozen)
    - tanh/sech² relative bias in attention (learnable, per head)
    """

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 d_ff=512, max_len=MAX_LEN, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # sin/cos absolute position encoding (frozen buffer)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, loss_mask=None):
        B, T = token_ids.shape
        device = token_ids.device

        # Token embedding + sin/cos absolute PE
        x = self.token_embedding(token_ids) + self.pe[:T, :]  # B,T,D

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # B,T,V

        loss = None
        if loss_mask is not None:
            # Standard next-token prediction:
            # logits[:, i, :] should predict token_ids[:, i+1]
            shift_logits = logits[:, :-1, :].contiguous()   # B, T-1, V
            shift_targets = token_ids[:, 1:].contiguous()    # B, T-1
            shift_mask = loss_mask[:, :-1].contiguous()       # B, T-1

            active_logits = shift_logits[shift_mask.bool()]
            active_targets = shift_targets[shift_mask.bool()]
            loss = F.cross_entropy(active_logits, active_targets)

        return logits, loss

    def init_weights(self):
        """Xavier init for Linear/Embedding, skip tanh params."""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if any(x in name for x in ['w_param', 'v_param', 'log_tau']):
                continue
            nn.init.xavier_uniform_(param)

# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, data, device, use_teacher_forcing=True):
    """Sample-level accuracy on answer tokens."""
    model.eval()
    correct = 0

    for sample in data:
        ids = encode(sample)
        eq_pos = ids.index(TOKEN2ID['='])
        answer_len = len(ids) - eq_pos - 1  # tokens after '=', incl EOS

        if use_teacher_forcing:
            # Feed full sequence; preds[i] should match ids[i+1]
            ids_tensor = torch.tensor([ids], device=device)
            logits, _ = model(ids_tensor, None)
            preds = logits[0].argmax(dim=-1)  # T predictions

            all_ok = True
            for i in range(eq_pos, len(ids) - 1):
                if preds[i] != ids[i + 1]:
                    all_ok = False
                    break
            if all_ok:
                correct += 1
        else:
            # Autoregressive: feed up to '=', generate rest
            prefix_ids = ids[:eq_pos + 1]  # up to and including '='
            gen_ids = list(prefix_ids)
            for _ in range(answer_len):
                ids_tensor = torch.tensor([gen_ids], device=device)
                logits, _ = model(ids_tensor, None)
                next_token = logits[0, -1].argmax(dim=-1).item()
                gen_ids.append(next_token)
                if next_token == TOKEN2ID['EOS']:
                    break
            if gen_ids == ids:
                correct += 1

    return correct / len(data)

# ============================================================
# TRAINING
# ============================================================
def train():
    device = torch.device('cpu')
    set_seed(SEED)

    print("=" * 60)
    print("verify3.2: Mixed PE (sin/cos + tanh/sech²)")
    print("=" * 60)

    # Data
    print(f"\nGenerating {TRAIN_SIZE:,} train + {VAL_SIZE:,} val samples...")
    train_raw = generate_dataset(TRAIN_SIZE)
    val_raw = generate_dataset(VAL_SIZE)

    # Encode all
    train_encoded = [encode(s) for s in train_raw]
    val_encoded = [encode(s) for s in val_raw]

    # Model
    model = MixedPETransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)
    model.init_weights()

    print(f"Total params:     {count_params(model):>8,}")
    print(f"Trainable params: {count_trainable(model):>8,}")
    pe_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(x in n for x in ['w_param', 'v_param', 'log_tau'])
    )
    print(f"  of which tanh:  {pe_params:>8,}")
    print(f"  frozen (PE):    {model.pe.numel():>8,}")

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>6} {'Loss':>8} {'ValAcc':>8} {'Best':>8} {'LR':>10}")
    print("-" * 45)

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0
        random.shuffle(train_encoded)

        for i in range(0, len(train_encoded), BATCH_SIZE):
            batch = train_encoded[i:i + BATCH_SIZE]
            token_ids, loss_mask = collate_batch(batch)
            token_ids = token_ids.to(device)
            loss_mask = loss_mask.to(device)

            _, loss = model(token_ids, loss_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ---- Eval ----
        val_acc = evaluate(model, val_raw, device, use_teacher_forcing=True)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Best tracking
        best_marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_marker = " ←"
            torch.save(model.state_dict(), 'best.pth')

        print(f"{epoch:>6} {avg_loss:>8.4f} {val_acc:>7.2%} {best_val_acc:>7.2%} "
              f"{current_lr:>10.2e}{best_marker}")

        # Early stop
        if val_acc >= EARLY_STOP_ACC:
            print(f"\n🎉 Early stop triggered! ValAcc {val_acc:.2%} ≥ {EARLY_STOP_ACC:.0%}")
            # Save checkpoint
            torch.save(model.state_dict(), 'best.pth')
            print(f"  Saved: best.pth")
            break

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best ValAcc:  {best_val_acc:.2%} @ epoch {best_epoch}")
    print(f"  Final epoch:  {epoch}")
    print(f"  Total params: {count_params(model):,}")

    # Quick autoregressive test
    print(f"\n--- Autoregressive eval (strict) ---")
    strict_acc = evaluate(model, val_raw, device, use_teacher_forcing=False)
    print(f"  Strict Acc:   {strict_acc:.2%}")

    return model, best_val_acc


if __name__ == '__main__':
    train()
