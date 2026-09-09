"""测试模型生成"""
import torch, sys, re
sys.path.insert(0, "/root/airesearch/v2")
sys.path.insert(0, "/root/airesearch/v3")
from v2 import CharTokenizer
from v3_grpo_chain import TransformerCoTWithCache, compute_step_reward

DEVICE = "cpu"
tokenizer = CharTokenizer()

model = TransformerCoTWithCache(
    vocab_size=tokenizer.vocab_size, d_model=128,
    n_layers=4, n_heads=4, d_ff=512, max_len=256, dropout=0.1
)

tests = [
    ("57-23+35=", 69),
    ("12-3+5=", 14),
    ("100+50-30=", 120),
    ("1+2=", 3),
    ("20-10=", 10),
]

for label, path in [
    ("Epoch 0 (SFT)", "/root/airesearch/v3/checkpoints/transformer_cot_opt_epoch0.pth"),
    ("Best (epoch 8)", "/root/airesearch/v3/checkpoints/transformer_cot_opt_best.pth"),
    ("Final (epoch 30)", "/root/airesearch/v3/checkpoints/transformer_cot_opt_latest.pth"),
]:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    print("\n" + "="*60)
    print(label)
    print("="*60)

    for prompt, gt in tests:
        resp, tokens = model.generate_with_cache(
            prompt, tokenizer, max_gen_len=60,
            temperature=0.6, top_p=0.9
        )
        rew = compute_step_reward(resp, gt, prompt)
        m = re.search(r"=(-?\d+)\s*$", resp)
        pred = m.group(1) if m else "?"
        ok = "OK" if rew >= 0.3 else " X"
        display = resp.replace("\n", "\\n")[:60]
        print(f"  [{ok}] {prompt:>15} -> {display}")
        print(f"        pred={pred:>6} gt={gt:>5} R={rew:+.2f}")
