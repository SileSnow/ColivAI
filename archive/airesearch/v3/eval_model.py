import sys
sys.path.insert(0, '/root/airesearch/v3')
sys.path.insert(0, '/root/airesearch/v2')

import torch
import random

# 导入模型和 tokenizer
from v3_grpo_chain import TransformerCoTWithCache, CharTokenizer
from v3_grpo_chain import generate_chain_cot, compute_step_reward

# 配置
CKPT = '/root/airesearch/v3/checkpoints/transformer_cot_opt_best.pth'
DEVICE = 'cpu'

# 加载 tokenizer
tokenizer = CharTokenizer()

# 构建模型
model = TransformerCoTWithCache(
    vocab_size=tokenizer.vocab_size,
    d_model=128, n_layers=4,
    n_heads=4, d_ff=512,
    max_len=256, dropout=0.0
)

# 加载权重
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()

print(f'✅ 模型加载成功 ({CKPT})')
print(f'   参数量: {sum(p.numel() for p in model.parameters()):,}')
print()

# 测试 prompt 列表
test_prompts = [
    '12+5=',
    '34-12=',
    '8+15-3=',
    '100-45+27=',
    '5+3-2+8=',
    '50+60-30+20-10=',
]

print('=' * 70)
print('🧪 测试生成结果（temperature=0.6）')
print('=' * 70)

for prompt in test_prompts:
    resp, tokens = model.generate_with_cache(
        prompt, tokenizer,
        max_gen_len=60,
        temperature=0.6,
        top_p=0.9
    )
    # 算奖励
    reward = compute_step_reward(resp, 0, prompt)
    print(f'Prompt: {prompt}')
    print(f'输出:   {repr(resp)}')
    print(f'-' * 70)

print()
print('=' * 70)
print('🧪 测试生成结果（temperature=0.3, 更确定）')
print('=' * 70)

for prompt in test_prompts:
    resp, tokens = model.generate_with_cache(
        prompt, tokenizer,
        max_gen_len=60,
        temperature=0.3,
        top_p=0.9
    )
    reward = compute_step_reward(resp, 0, prompt)
    print(f'Prompt: {prompt}')
    print(f'输出:   {repr(resp)}')
    print(f'-' * 70)

