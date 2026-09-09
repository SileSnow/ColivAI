import sys, torch, re, math
sys.path.insert(0, "/root/airesearch/v2")
sys.path.insert(0, "/root/airesearch/v3")
from v2 import CharTokenizer, PAD_IDX, BOS_IDX, EOS_IDX
import importlib.util

spec = importlib.util.spec_from_file_location("v3_zero", "/root/airesearch/v3/v3_zero.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
TransformerCoTWithCache = v3.TransformerCoTWithCache

device = "cpu"
tokenizer = CharTokenizer()

def load_model(ckpt_path, label):
    model = TransformerCoTWithCache(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_layers=4, n_heads=4, d_ff=512, max_len=256, dropout=0.0
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if all(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠️  missing keys: {missing}")
    model.eval()
    print(f"  [{label}] 已加载")
    return model

models = {
    "SFT": "/root/airesearch/v3/checkpoints_v3_zero/sft_best.pth",
    "GRPO_best": "/root/airesearch/v3/checkpoints_v3_zero/transformer_cot_v3_zero_best.pth",
    "GRPO_epoch25": "/root/airesearch/v3/checkpoints_v3_zero/transformer_cot_v3_zero_epoch25.pth",
}

test_prompts = [
    "12+34=",
    "45-18=",
    "3+5-2=",
    "100-50+25=",
    "-8+15-3=",
]

# 1) 贪婪解码对比
for label, ckpt in models.items():
    try:
        model = load_model(ckpt, label)
        print(f"\n{'='*50}")
        print(f"  {label}  (T=0.1, 接近贪婪)")
        print(f"{'='*50}")
        for p in test_prompts:
            resp, ids = model.generate_with_cache(p, tokenizer, max_gen_len=60, temperature=0.1, top_p=1.0)
            resp_show = resp.replace("\n", " | ")
            print(f"  P: {p}")
            print(f"  -> {resp_show}")
            print()
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        import traceback
        traceback.print_exc()

# 2) GRPO_best 多样采样 + 检测有效 step
print(f"\n{'='*50}")
print(f"  GRPO_best 多样采样 (T=0.8)")
print(f"{'='*50}")
try:
    model = load_model(models["GRPO_best"], "GRPO_best")
    step_re = re.compile(r'^(-?\d+)([+-])(-?\d+)=(-?\d+)$')
    for p in test_prompts[:3]:
        print(f"  P: {p}")
        for i in range(3):
            resp, ids = model.generate_with_cache(p, tokenizer, max_gen_len=50, temperature=0.8, top_p=0.9)
            resp_show = resp.replace("\n", " | ")
            print(f"    [{i}] {resp_show}")
            # 检查有效 step
            for s in resp.strip().split("\n"):
                s = s.strip()
                if step_re.match(s):
                    print(f"        ✅ 有效 step: {s}")
        print()
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ 评估完成")
