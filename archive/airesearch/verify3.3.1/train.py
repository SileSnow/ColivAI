"""
verify3.3.1: Mixed PE (non-zero tanh init) on Continuous Chain Arithmetic
================================================================
- 任务：连续链 ± 运算，2~8 步随机长度
- 模型：sin/cos 绝对 PE + tanh/sech² 相对偏置（同 verify3.2 架构）
- 目标：检验长序列下 tanh/sech² 是否真正学习，与短序列对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# CONFIG
# ============================================================
VOCAB = ['0','1','2','3','4','5','6','7','8','9','+','-','=','\n','PAD','BOS','EOS']
TOKEN2ID = {c: i for i, c in enumerate(VOCAB)}
ID2TOKEN = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)  # 17

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.0
MAX_LEN = 128

BATCH_SIZE = 16       # smaller batches for longer sequences (mem/speed)
LR = 5e-4              # reduced from 1e-3 for stability
EPOCHS = 300
EARLY_STOP_ACC = 0.95
WARMUP_EPOCHS = 5      # linear warmup

TRAIN_SIZE = 20000
VAL_SIZE = 2000
CHAIN_MIN = 2
CHAIN_MAX = 8

SEED = 42

# ============================================================
# UTILS
# ============================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ============================================================
# DATA: Continuous Chain Generation
# ============================================================
def generate_chain():
    """Generate a continuous chain and return (text, final_result)."""
    n_steps = random.randint(CHAIN_MIN, CHAIN_MAX)
    a = random.randint(1, 99)  # avoid leading zeros issues

    steps = []
    for _ in range(n_steps):
        op = random.choice(['+', '-'])
        b = random.randint(1, 99)
        c = a + b if op == '+' else a - b
        steps.append(f"{a}{op}{b}={c}")
        a = c  # chain result becomes next operand

    # Build full text: "step1\nstep2\n...\n=final"
    text = "\n".join(steps) + f"\n={a}"
    return text

def generate_dataset(size):
    return [generate_chain() for _ in range(size)]

def encode(text):
    """text → [BOS, ...chars..., EOS]"""
    ids = [TOKEN2ID['BOS']]
    for ch in text:
        ids.append(TOKEN2ID[ch])
    ids.append(TOKEN2ID['EOS'])
    return ids

def make_loss_mask(ids):
    """Mark prediction positions for the FINAL answer only.
    All '=' in chain steps are not in the loss.
    Only the last '=' (final answer marker) and following tokens are."""
    # Find the LAST '=' in the sequence
    last_eq = max(i for i, t in enumerate(ids) if t == TOKEN2ID['='])
    mask = [0] * len(ids)
    for i in range(last_eq, len(ids) - 1):
        mask[i] = 1
    return mask

def collate_batch(samples):
    """Pad to max length in batch."""
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
# MODEL (same architecture as verify3.2)
# ============================================================
class MixedAttention(nn.Module):
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

        self.w_param = nn.Parameter(torch.randn(nhead) * 0.1)
        self.v_param = nn.Parameter(torch.randn(nhead) * 0.02 + 0.01)
        self.log_tau = nn.Parameter(torch.zeros(nhead))
        self.max_len = max_len

    def forward(self, x, causal_mask):
        B, T, D = x.shape
        H = self.nhead
        dk = self.head_dim

        q = self.q_proj(x).view(B, T, H, dk).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dk).transpose(1, 2)

        scale = math.sqrt(dk)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Relative position bias (tanh + sech²)
        device = x.device
        pos = torch.arange(T, device=device, dtype=torch.float32)
        distances = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)

        w = self.w_param.view(H, 1, 1)
        v_p = self.v_param.view(H, 1, 1)
        tau = self.log_tau.exp().view(H, 1, 1)

        tanh_bias = w * torch.tanh(v_p * distances + 1e-8)
        # Clamp cosh input to prevent float32 overflow (cosh(85) ≈ 3e36 < 3.4e38)
        sech2_bias = 1.0 / torch.cosh(torch.clamp(tau * distances, max=85.0)) ** 2
        rel_bias = tanh_bias + sech2_bias

        attn_logits = attn_logits + rel_bias.unsqueeze(0)
        attn_logits = attn_logits.masked_fill(~causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights, v)
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
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 d_ff=512, max_len=MAX_LEN, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)

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

        x = self.token_embedding(token_ids) + self.pe[:T, :]
        causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if loss_mask is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = token_ids[:, 1:].contiguous()
            shift_mask = loss_mask[:, :-1].contiguous()

            active_logits = shift_logits[shift_mask.bool()]
            active_targets = shift_targets[shift_mask.bool()]
            loss = F.cross_entropy(active_logits, active_targets)

        return logits, loss

    def init_weights(self):
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
    """Sample-level accuracy on final answer."""
    model.eval()
    correct = 0

    for text in data:
        ids = encode(text)
        # Find where the final answer starts: last '=' in ids
        last_eq = max(i for i, t in enumerate(ids) if t == TOKEN2ID['='])
        # The answer tokens are ids[last_eq:] (including the '=' and final answer + EOS)
        answer_ids = ids[last_eq:]  # e.g., [=, 1, 3, EOS]
        answer_len = len(answer_ids) - 1  # predictions to generate (excluding last '=')

        if use_teacher_forcing:
            ids_tensor = torch.tensor([ids], device=device)
            logits, _ = model(ids_tensor, None)
            preds = logits[0].argmax(dim=-1)

            all_ok = True
            for i in range(last_eq, len(ids) - 1):
                if preds[i] != ids[i + 1]:
                    all_ok = False
                    break
            if all_ok:
                correct += 1
        else:
            # Autoregressive: feed up to last '=', generate answer
            prefix_ids = ids[:last_eq + 1]  # includes last '='
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
    print("verify3.3.1: Mixed PE (non-zero tanh init) on Continuous Chain (2-8 steps)")
    print("=" * 60)

    print(f"\nGenerating {TRAIN_SIZE:,} train + {VAL_SIZE:,} val chains...")
    train_raw = generate_dataset(TRAIN_SIZE)
    val_raw = generate_dataset(VAL_SIZE)

    # Show sample
    print(f"  Sample chain ({len(encode(train_raw[0]))} tokens):")
    print(f"    {repr(train_raw[0])}")

    train_encoded = [encode(s) for s in train_raw]
    val_encoded = [encode(s) for s in val_raw]

    lengths = [len(e) for e in train_encoded + val_encoded]
    print(f"  Seq length: min={min(lengths)}  max={max(lengths)}  avg={sum(lengths)/len(lengths):.1f}")

    model = MixedPETransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, d_ff=D_FF, max_len=MAX_LEN, dropout=DROPOUT,
    ).to(device)
    model.init_weights()

    print(f"\nTotal params:     {count_params(model):>8,}")
    tanh_params = sum(p.numel() for n, p in model.named_parameters()
                      if any(x in n for x in ['w_param', 'v_param', 'log_tau']))
    print(f"  of which tanh:  {tanh_params:>8,}")
    print(f"  frozen (PE):    {model.pe.numel():>8,}")

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>6} {'Loss':>8} {'ValAcc':>8} {'Best':>8} {'LR':>10}")
    print("-" * 45)

    for epoch in range(1, EPOCHS + 1):
        # Warmup
        if epoch <= WARMUP_EPOCHS:
            lr = LR * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            scheduler.step()

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

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠ NaN/Inf loss at batch {n_batches}, skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_acc = evaluate(model, val_raw, device, use_teacher_forcing=True)
        current_lr = optimizer.param_groups[0]['lr']

        best_marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_marker = " ←"

        print(f"{epoch:>6} {avg_loss:>8.4f} {val_acc:>7.2%} {best_val_acc:>7.2%} "
              f"{current_lr:>10.2e}{best_marker}")

        if val_acc >= EARLY_STOP_ACC:
            print(f"\n🎉 Early stop! ValAcc {val_acc:.2%}")
            torch.save(model.state_dict(), 'best.pth')
            break

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best ValAcc:  {best_val_acc:.2%} @ epoch {best_epoch}")
    print(f"  Final epoch:  {epoch}")

    strict_acc = evaluate(model, val_raw, device, use_teacher_forcing=False)
    print(f"  Strict Acc:   {strict_acc:.2%}")

    # tanh analysis
    print(f"\n--- tanh/sech² parameters ---")
    for layer_idx in range(NUM_LAYERS):
        attn = model.layers[layer_idx].attn
        active = 0
        for h in range(NHEAD):
            w = attn.w_param[h].item()
            v = attn.v_param[h].item()
            if abs(w) > 0.001 or abs(v) > 0.001:
                active += 1
        print(f"  Layer {layer_idx}: {active}/{NHEAD} heads active (|w|>0.001 or |v|>0.001)")

    return model, best_val_acc


if __name__ == '__main__':
    train()
