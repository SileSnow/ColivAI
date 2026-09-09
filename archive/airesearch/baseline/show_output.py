import torch, sys, random
sys.path.insert(0, '/root/airesearch/baseline')
import baseline
from baseline import SimpleTokenizer, StandardTransformer

tokenizer = SimpleTokenizer()
model = StandardTransformer(16, 128, 4, 4, 512, 32)
state = torch.load('/workspace/baseline/checkpoints/best.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

rng = random.Random(42)
tests = []
for _ in range(25):
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    op = rng.choice(['+', '-'])
    if op == '+': ans = a + b
    else: ans = a - b
    tests.append((f'{a}{op}{b}=', ans))

correct = 0
for prompt, gt in tests:
    prompt_ids = [tokenizer.BOS_IDX] + tokenizer.encode(prompt, add_special=False)
    gen = model.generate(prompt_ids, tokenizer, max_new_tokens=10, temperature=0)
    pred_text = tokenizer.decode(gen)
    eq = pred_text.find('=')
    out = pred_text[eq+1:] if eq >= 0 else '?'
    try:
        import re
        m = re.search(r'-?\d+', out)
        pred_val = int(m.group()) if m else None
    except:
        pred_val = None
    ok = 'OK' if pred_val == gt else 'XX'
    if pred_val == gt: correct += 1
    print(f'  [{ok}]  {prompt:>10}  pred={pred_val:>5}  gt={gt:>5}')

print(f'  Total: {correct}/{len(tests)}')
