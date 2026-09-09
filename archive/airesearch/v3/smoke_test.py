import sys; sys.path.insert(0,'/root/airesearch/v3')
from v3_fix import Tokenizer22, Transformer, gen_data, make_cot
import torch, torch.nn.functional as F, random

tok = Tokenizer22()
print('=== 1. Tokenizer 往返测试 ===')
for t in ['12+5=', '-3-7+2=', '100-45+27=', '5+3-2+8=']:
    ids = tok.encode(t, add_special=True)
    dec = tok.decode(ids)
    print(f'  {t:15s} -> ids={ids} -> {repr(dec)}')
    assert t in dec, f'Roundtrip failed: {t} not in {dec}'
print('  Tokenizer OK')

print('\n=== 2. 数据格式测试 ===')
data = gen_data(5, max_steps=3, max_val=50)
for p, tgt, ans in data[:2]:
    full = p+'\n'+tgt
    ids = tok.encode(full)
    prompt_ids = tok.encode(p, add_special=True)
    print(f'  prompt={p}, ans={ans}')
    print(f'  target={tgt[:40]}...')
    print(f'  full_ids len={len(ids)}, prompt_ids len={len(prompt_ids)}')
    # 验证: full 中 prompt 后面紧接 target, 无多余 EOS
    start = len(prompt_ids) - 1
    target_slice = ids[start:len(ids)-1]
    target_decoded = tok.decode(target_slice)
    assert '\n' in target_decoded, f'No newline in target: {target_decoded}'
print('  Data format OK')

print('\n=== 3. 小规模训练+生成测试 ===')
data = gen_data(200, max_steps=3, max_val=50)
random.shuffle(data)
train_d = data[:150]
val_d = data[150:]

m = Transformer(vocab=tok.vocab_size)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

print('  Training 5 epochs on 150 samples...')
for ep in range(1, 6):
    m.train(); random.shuffle(train_d)
    for i in range(0, len(train_d), 16):
        batch = train_d[i:i+16]
        all_ids, max_len = [], 0
        for prompt, target, _ in batch:
            ids = tok.encode(prompt+'\n'+target)
            all_ids.append(ids); max_len = max(max_len, len(ids))
        inp = torch.full((len(batch), max_len), 0, dtype=torch.long)
        lm = torch.zeros(len(batch), max_len)
        for j, ids in enumerate(all_ids):
            inp[j,:len(ids)] = torch.tensor(ids)
            pl = len(tok.encode(batch[j][0], add_special=True))
            lm[j, pl-1:len(ids)-1] = 1.0
        logits = m(inp)
        sl = logits[:,:-1,:].contiguous(); st = inp[:,1:].contiguous()
        raw = F.cross_entropy(sl.reshape(-1,tok.vocab_size), st.reshape(-1), ignore_index=0, reduction='none')
        loss = (raw * lm[:,:-1].reshape(-1)).sum() / lm[:,:-1].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 2 == 0:
        print(f'  Epoch {ep} loss={loss.item():.4f}')

print('\n  === 生成测试 ===')
m.eval()
for prompt, _, ans in val_d[:5]:
    # 关键：不带尾EOS
    prompt_ids = [1] + tok.encode(prompt, add_special=False)
    gen = m.generate(prompt_ids, temp=0.3)
    text = tok.decode(gen).replace('[BOS]','').replace('[EOS]','')
    # 解析答案
    eq_pos = [p for p,tid in enumerate(gen) if tid==15]
    pred = '?'
    if eq_pos:
        nums=[]
        for tid in gen[eq_pos[-1]+1:]:
            if 4<=tid<=13 or tid==14: nums.append(tok.i2c.get(tid,''))
            else: break
        try: pred=str(int(''.join(nums))) if nums else '?'
        except: pred='err'
    ok = 'OK' if str(pred)==str(ans) else 'WRONG'
    print(f'  [{prompt}] -> {repr(text)} ({ok} pred={pred} ans={ans})')

print('\n=== 4. 验证: 生成时 prompt 无尾EOS ===')
p = '12+5='
ids = [1] + tok.encode(p, add_special=False)
print(f'  prompt_ids={ids}')
print(f'  decoded={repr(tok.decode(ids))}')
assert ids[-1] != 2, f'BUG: prompt still has EOS! {ids}'
print('  No trailing EOS - OK')

print('\n✅ ALL CHECKS PASSED')
