"""
tiny — 缩小模型 SFT 实验
=========================
A 组 (frozen): 冻结预训练 embedding
B 组 (baseline): 随机初始化 embedding

架构: d_model=48, layers=2, heads=2, d_ff=96 (~38K params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math, random, time, os, re, sys

# ============================================================
class Config:
    D_MODEL   = 48
    N_LAYERS  = 2
    N_HEADS   = 2
    D_FF      = 96
    MAX_LEN   = 32
    DROPOUT   = 0.0

    BATCH_SIZE   = 64
    LR           = 1e-3
    EPOCHS       = 200
    EARLY_STOP_ACC = 0.99

    MIN_VAL   = 0
    MAX_VAL   = 99
    TRAIN_SAMPLES = 20000
    VAL_SAMPLES   = 2000

    CKPT_DIR = "/root/airesearch/small/checkpoints"
    LOG_FILE = "/root/airesearch/small/train.log"
    EMBED_PATH = "/root/airesearch/small/embedding.pth"

    DEVICE = "cpu"

# ============================================================
# Tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.specials = ['[PAD]', '[BOS]', '[EOS]']
        self.chars = list('0123456789+-=')
        self.all_tokens = self.specials + self.chars
        self.stoi = {ch: i for i, ch in enumerate(self.all_tokens)}
        self.itos = {i: ch for i, ch in enumerate(self.all_tokens)}
        self.PAD_IDX = self.stoi['[PAD]']
        self.BOS_IDX = self.stoi['[BOS]']
        self.EOS_IDX = self.stoi['[EOS]']

    @property
    def vocab_size(self):
        return len(self.all_tokens)

    def encode(self, text, add_special=True):
        tokens = [self.stoi[ch] for ch in text if ch in self.stoi]
        if add_special:
            tokens = [self.BOS_IDX] + tokens + [self.EOS_IDX]
        return tokens

    def decode(self, ids):
        return ''.join(self.itos.get(i, '?') for i in ids
                       if i not in (self.PAD_IDX, self.BOS_IDX, self.EOS_IDX))

# ============================================================
# SinCos positional encoding
class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

# ============================================================
# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# ============================================================
# Transformer
class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinCosPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'token_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def load_pretrained_embedding(self, path):
        pretrained = torch.load(path, map_location='cpu')
        self.token_embed.weight.data.copy_(pretrained)
        self.token_embed.weight.requires_grad = False
        print(f"Loaded pretrained embedding: {path} (frozen)")

    def forward(self, input_ids, return_logits=True):
        B, T = input_ids.shape
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=input_ids.device), diagonal=1)
        x = self.token_embed(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.ln_final(x)
        return self.output_proj(x)

    @torch.no_grad()
    def generate(self, prompt_ids, tokenizer, max_new_tokens=10, temperature=0.0):
        self.eval()
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            if len(generated) > self.max_len:
                break
            inp = torch.tensor([generated[-self.max_len:]], device=next(self.parameters()).device)
            logits = self.forward(inp)
            next_logits = logits[0, -1, :]
            if temperature == 0:
                next_id = next_logits.argmax().item()
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
            if next_id == tokenizer.EOS_IDX:
                break
        self.train()
        return generated

# ============================================================
# Dataset
class ArithmeticDataset(Dataset):
    def __init__(self, tokenizer, n_samples, min_val, max_val, seed=None):
        self.samples = []
        rng = random.Random(seed)
        for _ in range(n_samples):
            a = rng.randint(min_val, max_val)
            b = rng.randint(min_val, max_val)
            op = rng.choice(['+', '-'])
            result = a + b if op == '+' else a - b
            text = f"{a}{op}{b}={result}"
            ids = tokenizer.encode(text, add_special=True)
            self.samples.append((ids, result, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_idx):
    max_len = max(len(item[0]) for item in batch)
    padded, results, texts = [], [], []
    for ids, result, text in batch:
        pad_len = max_len - len(ids)
        padded.append(ids + [pad_idx] * pad_len)
        results.append(result)
        texts.append(text)
    return torch.tensor(padded, dtype=torch.long), results, texts

# ============================================================
# Accuracy
def compute_accuracy(model, dataset, tokenizer, config, n_eval=200):
    model.eval()
    correct = 0
    total = min(n_eval, len(dataset))
    indices = random.sample(range(len(dataset)), total)
    for idx in indices:
        ids, gt_result, text = dataset[idx]
        decoded = tokenizer.decode(ids)
        eq_pos = decoded.find('=')
        prompt_text = decoded[:eq_pos+1]
        prompt_ids = [tokenizer.BOS_IDX] + tokenizer.encode(prompt_text, add_special=False)
        generated = model.generate(prompt_ids, tokenizer, max_new_tokens=10, temperature=0)
        pred_text = tokenizer.decode(generated)
        eq_idx = pred_text.find('=')
        if eq_idx != -1:
            pred_str = pred_text[eq_idx+1:].strip()
            match = re.search(r'-?\d+', pred_str)
            if match:
                try:
                    if int(match.group()) == gt_result:
                        correct += 1
                except:
                    pass
    model.train()
    return correct / total if total > 0 else 0.0

# ============================================================
def train(use_frozen_embedding=False):
    config = Config()
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    tokenizer = SimpleTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_ds = ArithmeticDataset(tokenizer, config.TRAIN_SAMPLES, config.MIN_VAL, config.MAX_VAL, seed=42)
    val_ds = ArithmeticDataset(tokenizer, config.VAL_SAMPLES, config.MIN_VAL, config.MAX_VAL, seed=123)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer.PAD_IDX))
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer.PAD_IDX))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = StandardTransformer(tokenizer.vocab_size, config.D_MODEL, config.N_LAYERS,
                                 config.N_HEADS, config.D_FF, config.MAX_LEN, config.DROPOUT).to(config.DEVICE)

    if use_frozen_embedding:
        model.load_pretrained_embedding(config.EMBED_PATH)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config.LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)

    group_name = "frozen" if use_frozen_embedding else "baseline"
    print(f"\n{'='*60}")
    print(f"tiny experiment — {group_name} ({n_total:,} params)")
    print(f"{'='*60}\n")

    best_acc = 0.0
    log_lines = []

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_ids, _, _ in train_loader:
            batch_ids = batch_ids.to(config.DEVICE)
            B, T = batch_ids.shape
            logits = model(batch_ids)
            logits = logits[:, :-1, :].contiguous()
            targets = batch_ids[:, 1:].contiguous()

            eq_id = tokenizer.stoi['=']
            masked_targets = targets.clone()
            for b in range(B):
                eq_positions = (batch_ids[b] == eq_id).nonzero(as_tuple=True)[0]
                if len(eq_positions) > 0:
                    eq_pos = eq_positions[0].item()
                    target_start = max(0, eq_pos - 1)
                    if target_start < masked_targets.shape[1]:
                        masked_targets[b, :target_start] = tokenizer.PAD_IDX

            loss = loss_fn(logits.view(-1, tokenizer.vocab_size), masked_targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0

        train_acc = compute_accuracy(model, train_ds, tokenizer, config, n_eval=100)
        val_acc = compute_accuracy(model, val_ds, tokenizer, config, n_eval=100)

        lr_now = scheduler.get_last_lr()[0]
        line = (f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"TrainAcc: {train_acc:.1%} | ValAcc: {val_acc:.1%} | "
                f"LR: {lr_now:.2e} | {elapsed:.1f}s")
        print(line, flush=True)
        log_lines.append(line)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config.CKPT_DIR}/best_{group_name}.pth")
            print(f"  * new best! ValAcc={val_acc:.1%}", flush=True)

        if val_acc >= config.EARLY_STOP_ACC:
            print(f"\nearly stop! ValAcc {val_acc:.1%} >= {config.EARLY_STOP_ACC:.0%}")
            break

    # Final eval
    best_path = f"{config.CKPT_DIR}/best_{group_name}.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=config.DEVICE, weights_only=True))
    model.eval()
    final_acc = compute_accuracy(model, val_ds, tokenizer, config, n_eval=500)
    print(f"\nFinal accuracy (500 samples): {final_acc:.1%}")

    print(f"\nSamples:")
    for _ in range(8):
        idx = random.randint(0, len(val_ds) - 1)
        ids, gt, text = val_ds[idx]
        decoded = tokenizer.decode(ids)
        eq_pos = decoded.find('=')
        prompt_text = decoded[:eq_pos+1]
        prompt_ids = [tokenizer.BOS_IDX] + tokenizer.encode(prompt_text, add_special=False)
        generated = model.generate(prompt_ids, tokenizer, max_new_tokens=10, temperature=0)
        pred_text = tokenizer.decode(generated)
        ok = "OK" if str(gt) in pred_text else "FAIL"
        eq_in_pred = pred_text.find('=') if '=' in pred_text else len(pred_text)
        pred_ans = pred_text[eq_in_pred+1:].strip() if eq_in_pred < len(pred_text) else '?'
        print(f"  {ok} {text[:text.find('=')]}=  -> pred={pred_ans}  (gt={gt})")

    with open(config.LOG_FILE, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog: {config.LOG_FILE}")
    torch.save(model.state_dict(), f"{config.CKPT_DIR}/final_{group_name}.pth")
    print(f"Model: {config.CKPT_DIR}/")

    return model, tokenizer, final_acc, log_lines

if __name__ == "__main__":
    use_frozen = "--frozen" in sys.argv
    mode = "FROZEN pretrained embedding" if use_frozen else "RANDOM embedding (baseline)"
    print("=" * 60)
    print(f"tiny experiment — {mode}")
    print(f"d_model=48, layers=2, heads=2, d_ff=96")
    print("=" * 60)
    model, tokenizer, acc, logs = train(use_frozen_embedding=use_frozen)
    print(f"\nDone! Final accuracy: {acc:.1%}")
