"""
v3_fix.py — V3 修复版 v2：链式算术，纯 SFT + 自回归验证

Bug修复:
  1. 生成时 prompt 去掉尾 EOS（训练数据中 prompt 后紧跟 \n，没有 EOS）
  2. Epoch 5 强制存档，后续仅 AR-Acc 提升时存
  3. 抽取答案直接从 token ids 解析，避免 [EOS] 字符串干扰
"""
import torch, torch.nn as nn, torch.nn.functional as F
import random, math, time, os, sys

PAD, BOS, EOS = 0, 1, 2

# ═══ Tokenizer ═══
class Tokenizer22:
    def __init__(self):
        chars = list('0123456789+-=\n') + [' ', '*', '/', '(', ')']
        self.c2i = {c: i+3 for i, c in enumerate(chars)}
        self.c2i['[PAD]'] = PAD; self.c2i['[BOS]'] = BOS; self.c2i['[EOS]'] = EOS
        self.i2c = {v: k for k, v in self.c2i.items()}
    @property
    def vocab_size(self): return len(self.c2i)
    def encode(self, text, add_special=True):
        ids = [BOS] if add_special else []
        for c in text: ids.append(self.c2i.get(c, 3))
        if add_special: ids.append(EOS)
        return ids
    def decode(self, ids):
        return ''.join(self.i2c.get(i, '?') for i in ids)

# ═══ 数据 ═══
def make_one(start, ops, nums):
    line = str(start)
    for op, n in zip(ops, nums): line += f'{op}{n}'
    line += '='
    cur = start
    steps = []
    for op, n in zip(ops, nums):
        cur = cur + n if op == '+' else cur - n
        steps.append(f'{cur - n if op == "+" else cur + n}{op}{n}={cur}')
    steps.append(f'={cur}')
    return line, '\n'.join(steps), cur

# 修正版 make_one（直接正确计算）
def make_cot(start, ops, nums):
    prompt = str(start)
    for op, n in zip(ops, nums): prompt += f'{op}{n}'
    prompt += '='
    cur = start
    lines = []
    for op, n in zip(ops, nums):
        prev = cur
        cur = cur + n if op == '+' else cur - n
        lines.append(f'{prev}{op}{n}={cur}')
    lines.append(f'={cur}')
    return prompt, '\n'.join(lines), cur

def gen_data(n=20000, min_steps=2, max_steps=6, max_val=200):
    data = []
    for _ in range(n):
        steps = random.randint(min_steps, max_steps)
        start = random.randint(-50, max_val)
        ops = [random.choice(['+', '-']) for _ in range(steps)]
        nums = [random.randint(1, max_val) for _ in range(steps)]
        prompt, target, ans = make_cot(start, ops, nums)
        data.append((prompt, target, ans))
    return data

# ═══ 模型 ═══
class Transformer(nn.Module):
    def __init__(self, vocab=22, d_model=128, n_layers=4, n_heads=4, d_ff=512, max_len=256):
        super().__init__()
        self.d_model = d_model; self.max_len = max_len
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, 0, d_model**-0.5)

    def forward(self, x):
        B, L = x.shape
        h = self.embed(x) + self.pos[:, :L, :]
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), 1)
        return self.lm_head(self.encoder(h, mask=mask))

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=80, temp=0.5):
        self.eval()
        device = next(self.parameters()).device
        gen = list(prompt_ids)
        for _ in range(max_new):
            ctx = gen[-self.max_len:]
            inp = torch.tensor([ctx], device=device)
            logits = self(inp)
            probs = torch.softmax(logits[0, -1, :] / temp, dim=-1)
            tok = torch.multinomial(probs, 1).item()
            gen.append(tok)
            if tok == EOS: break
        return gen

# ═══ 训练 ═══
def train():
    tok = Tokenizer22()
    print(f'[Tokenizer] {tok.vocab_size} tokens')

    print('[Data] generating 20000 samples...')
    data = gen_data(20000)
    random.shuffle(data)
    split = int(len(data) * 0.9)
    train_set, val_set = data[:split], data[split:]
    print(f'[Data] train={len(train_set)}, val={len(val_set)}')

    m = Transformer(vocab=tok.vocab_size)
    print(f'[Model] {sum(p.numel() for p in m.parameters()):,} params')

    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    best_acc, bs = 0.0, 32

    for epoch in range(1, 51):
        m.train(); random.shuffle(train_set)
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(train_set), bs):
            batch = train_set[i:i+bs]
            # pack & pad
            all_ids, max_len = [], 0
            for prompt, target, _ in batch:
                ids = tok.encode(prompt + '\n' + target)
                all_ids.append(ids)
                max_len = max(max_len, len(ids))

            inp = torch.full((len(batch), max_len), PAD, dtype=torch.long)
            loss_mask = torch.zeros(len(batch), max_len)
            for j, ids in enumerate(all_ids):
                inp[j, :len(ids)] = torch.tensor(ids)
                # prompt 编码长度（含BOS/EOS）= prompt字符数 + 2
                # 在完整序列中，prompt后紧跟 \n + target，无EOS
                # 所以 loss 从 prompt 最后一个字符之后开始
                prompt_ids = tok.encode(batch[j][0], add_special=True)
                start = len(prompt_ids) - 1  # prompt的EOS位置 = target第一个字符位置
                loss_mask[j, start:len(ids)-1] = 1.0  # 到末尾EOS之前

            logits = m(inp)
            sl = logits[:, :-1, :].contiguous()
            st = inp[:, 1:].contiguous()
            raw = F.cross_entropy(sl.reshape(-1, tok.vocab_size), st.reshape(-1),
                                  ignore_index=PAD, reduction='none')
            mask_flat = loss_mask[:, :-1].reshape(-1)
            loss = (raw * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f'[Epoch {epoch:2d}] loss={avg_loss:.4f}', end='')

        # ── 自回归评估 ──
        if epoch % 5 == 0 or epoch == 1:
            m.eval()
            correct, n_eval = 0, min(200, len(val_set))
            samples_shown = 0
            for prompt, _, ans in val_set[:n_eval]:
                # Bug 1 修复: prompt 不带尾 EOS
                prompt_ids = [BOS] + tok.encode(prompt, add_special=False)
                gen = m.generate(prompt_ids, temp=0.5)
                # 从 token ids 直接解析答案: 最后一个 = 之后的连续数字
                eq_pos = [p for p, tid in enumerate(gen) if tid == 15]  # '=' = id 15
                if eq_pos:
                    num_ids = []
                    for tid in gen[eq_pos[-1]+1:]:
                        if 4 <= tid <= 13 or tid == 14:  # 0-9 or -
                            num_ids.append(tok.i2c.get(tid, ''))
                        else: break
                    try:
                        if num_ids and int(''.join(num_ids)) == ans:
                            correct += 1
                    except: pass
                # 打印前3个示例
                if samples_shown < 3:
                    text = tok.decode(gen).replace('[BOS]','').replace('[EOS]','')
                    print(f'\n  [{prompt}] -> {repr(text)} (ans={ans})', end='')
                    samples_shown += 1

            ar_acc = correct / n_eval * 100
            print(f'\n[Eval] AR-Acc={ar_acc:.1f}%', end='')
            if ar_acc > best_acc:
                best_acc = ar_acc
                torch.save(m.state_dict(), '/root/airesearch/v3/checkpoints/v3_fix_best.pth')
                print(' (saved)', end='')
            # Bug 2 修复: epoch 5 强制存档
            if epoch == 5:
                torch.save(m.state_dict(), '/root/airesearch/v3/checkpoints/v3_fix_epoch5.pth')
                print(' (e5 saved)', end='')
        print()

        if ar_acc >= 95.0:
            print(f'[Done] Reached 95% at epoch {epoch}!')
            break

    print(f'[Done] Best AR-Acc: {best_acc:.1f}%')

if __name__ == '__main__':
    train()
