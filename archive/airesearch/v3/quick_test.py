import sys; sys.path.insert(0,'/root/airesearch/v3')
from v3_fix import Tokenizer22, SimpleTransformer, gen_data
import torch, random

tok = Tokenizer22()
# 用小数据集快速测试
data = gen_data(2000, max_steps=4, max_val=100)
random.shuffle(data)
split = int(len(data)*0.9)
train_data, val_data = data[:split], data[split:]
print(f'Train: {len(train_data)}, Val: {len(val_data)}')

m = SimpleTransformer(vocab=tok.vocab_size)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
bs = 32

for epoch in range(1, 16):
    m.train(); random.shuffle(train_data)
    for i in range(0, len(train_data), bs):
        batch = train_data[i:i+bs]
        all_ids = []; max_len = 0
        for prompt, target, _ in batch:
            full = prompt + '\n' + target
            ids = tok.encode(full); all_ids.append(ids)
            max_len = max(max_len, len(ids))
        padded = torch.full((len(batch), max_len), 0, dtype=torch.long)
        loss_mask = torch.zeros(len(batch), max_len)
        for j, ids in enumerate(all_ids):
            padded[j,:len(ids)] = torch.tensor(ids)
            prompt_len = len(tok.encode(batch[j][0], add_special=True))
            loss_mask[j, prompt_len-1:len(ids)-1] = 1.0
        logits = m(padded)
        sl = logits[:,:-1,:].contiguous(); st = padded[:,1:].contiguous()
        raw = torch.nn.functional.cross_entropy(sl.reshape(-1,tok.vocab_size), st.reshape(-1), ignore_index=0, reduction='none')
        mf = loss_mask[:,:-1].reshape(-1)
        loss = (raw * mf).sum() / mf.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    
    if epoch % 5 == 0:
        m.eval(); correct = 0
        for prompt, target, ans in val_data[:100]:
            prompt_ids = tok.encode(prompt, add_special=True)
            gen_ids = m.generate(prompt_ids, temp=0.3)
            try:
                eq_pos = [i for i, tid in enumerate(gen_ids) if tid == 15]
                if eq_pos:
                    nums = []
                    for tid in gen_ids[eq_pos[-1]+1:]:
                        if 4 <= tid <= 13 or tid == 14: nums.append(tok.i2c.get(tid,''))
                        else: break
                    if nums and int(''.join(nums)) == ans: correct += 1
            except: pass
        ar = correct / 100 * 100
        print(f'Epoch {epoch:2d} | AR-Acc={ar:.0f}%')
        # 打印示例
        for prompt, target, ans in val_data[:3]:
            prompt_ids = tok.encode(prompt, add_special=True)
            gen_ids = m.generate(prompt_ids, temp=0.3)
            text = tok.decode(gen_ids).replace('[BOS]','').replace('[EOS]','')
            print(f'  {prompt} -> {repr(text)} (true={ans})')
