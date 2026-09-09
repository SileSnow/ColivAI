import torch, sys
sys.path.insert(0, '/root/airesearch/baseline')
from baseline import SimpleTokenizer, StandardTransformer, Config

config = Config()
tokenizer = SimpleTokenizer()
model = StandardTransformer(16, 128, 4, 4, 512, 32)
model.load_state_dict(torch.load('/root/airesearch/baseline/checkpoints/best.pth', map_location='cpu', weights_only=True))
model.eval()

tests = ['12+34=', '99-51=', '7+8=', '50-25=']
for t in tests:
    prompt_ids = [tokenizer.BOS_IDX] + tokenizer.encode(t, add_special=False)
    gen = model.generate(prompt_ids, tokenizer, max_new_tokens=10, temperature=0)
    print(f'{t} -> {tokenizer.decode(gen)}')
