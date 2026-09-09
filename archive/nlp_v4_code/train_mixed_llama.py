"""
V4-Mixed-LLaMA-103 — LLaMA架构 + tanh/sech² PE + WikiText-103
================================================================
"""
import os, math, time, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import sentencepiece as spm
from model_mixed_llama import TransformerLM_MixedLLaMA

# ─── 配置 ──────────────────────────────────────────────────
DATA_DIR     = "/root/nlp_v4/data/wiki103"
MODEL_DIR    = "/root/nlp_v4/models"
LOG_DIR      = "/root/nlp_v4/logs"
BPE_PREFIX   = DATA_DIR + "/spm_bpe_5000"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

VOCAB_SIZE   = 5000
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SEQ_LEN      = 512
D_MODEL      = 256
NHEAD        = 4
NLAYER       = 6
D_INTER      = 688        # SwiGLU intermediate (matches FFN param count)
MAX_LEN      = 600
DROPOUT      = 0.0

BATCH_SIZE   = 32
SUBSET_TRAIN = 50000     # 只用前 50K 块（~9%），平衡速度与数据量
EPOCHS       = 2
LR           = 3e-4
LR_MIN       = 1e-5
WARMUP_STEPS = 500        # 按步数热身
GRAD_CLIP    = 1.0
GEN_EVERY    = 2000       # 按步数生成演示
EVAL_EVERY   = 1000       # 按步数评估

# ─── 数据集 ──────────────────────────────────────────────────

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.inputs  = chunks[:, :-1]
        self.targets = chunks[:, 1:]
    def __len__(self): return len(self.inputs)
    def __getitem__(self, i): return self.inputs[i], self.targets[i]

# ─── 训练 ──────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 数据
    train_chunks = torch.load(DATA_DIR + "/train.pt", weights_only=True)[:SUBSET_TRAIN]
    valid_chunks = torch.load(DATA_DIR + "/valid.pt", weights_only=True)
    print(f"[Data] train={train_chunks.shape} valid={valid_chunks.shape}")
    train_loader = DataLoader(LMDataset(train_chunks), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(LMDataset(valid_chunks), batch_size=BATCH_SIZE)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"[Steps] {steps_per_epoch}/epoch × {EPOCHS} = {total_steps} total")

    # 模型
    model = TransformerLM_MixedLLaMA(VOCAB_SIZE, D_MODEL, NHEAD, NLAYER, D_INTER, MAX_LEN, DROPOUT, PAD_ID).to(device)
    print(f"[Model] {model.count_parameters():,} params (LLaMA+tanh/sech2)")

    # 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    sp = spm.SentencePieceProcessor(); sp.load(BPE_PREFIX + ".model")

    best_val_ppl = float("inf")
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (inp, tgt) in enumerate(train_loader):
            inp, tgt = inp.to(device), tgt.to(device)

            # LR schedule: warmup + cosine
            if global_step < WARMUP_STEPS:
                lr = LR * global_step / WARMUP_STEPS
            else:
                progress = (global_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
                lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
            for pg in opt.param_groups: pg["lr"] = lr

            opt.zero_grad()
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1), ignore_index=PAD_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            global_step += 1

            if global_step % 50 == 0 or global_step == 1:
                ppl = math.exp(loss.item())
                elapsed = time.time() - t0
                print(f"[Step {global_step:6d}/{total_steps}] loss={loss.item():.4f} ppl={ppl:.1f} lr={lr:.2e} {elapsed:.0f}s")

            # 定期评估
            if global_step % EVAL_EVERY == 0:
                model.eval()
                val_loss = 0.0; val_tokens = 0
                with torch.no_grad():
                    for vi, vt in valid_loader:
                        vi, vt = vi.to(device), vt.to(device)
                        vl = F.cross_entropy(model(vi).reshape(-1, VOCAB_SIZE), vt.reshape(-1), ignore_index=PAD_ID)
                        val_loss += vl.item() * vi.numel(); val_tokens += vi.numel()
                val_ppl = math.exp(val_loss / val_tokens)
                print(f"  [Eval  @ step {global_step}] Val PPL: {val_ppl:.2f}")
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    torch.save(model.state_dict(), MODEL_DIR + "/best_mixed_llama103.pt")
                    print(f"  [Save] best Val PPL: {val_ppl:.2f}")
                model.train()

            # 定期生成
            if global_step % GEN_EVERY == 0:
                model.eval()
                prompt = "The "
                ids = [BOS_ID] + sp.encode(prompt, out_type=int)
                gen = model.generate(ids, max_new_tokens=50, temperature=0.8)
                print(f"  [Gen] {sp.decode(gen)}")
                model.train()

        print(f"[Epoch {epoch} done] {time.time()-t0:.0f}s")

    print(f"[Done] Best Val PPL: {best_val_ppl:.2f}")

    # 保存 tanh/sech² 参数快照
    pe_snapshot = {}
    for l, layer in enumerate(model.layers):
        attn = layer.self_attn
        pe_snapshot[f"L{l}"] = {
            "w": attn.w_param.detach().tolist(),
            "v": attn.v_param.detach().tolist(),
            "tau": torch.exp(attn.log_tau).detach().tolist(),
        }
    with open(LOG_DIR + "/mixed_llama103_pe_params.json", "w") as f:
        json.dump(pe_snapshot, f, indent=2)
    print(f"[PE params saved] {LOG_DIR}/mixed_llama103_pe_params.json")

if __name__ == "__main__":
    main()
