"""分析 checkpoint：对比 epoch0 → best → final"""
import torch
import sys
import numpy as np
sys.path.insert(0, '/root/airesearch/v2')
from v2 import CharTokenizer

CKPT_DIR = "/root/airesearch/v3/checkpoints"

def load_sd(name):
    path = f"{CKPT_DIR}/transformer_cot_opt_{name}.pth"
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        return sd, ckpt
    return ckpt, {}

# 1. 基本信息
print("="*60)
print("📋 CHECKPOINT 基本信息")
print("="*60)

for name, label in [("epoch0", "Epoch 0 (SFT后)"), 
                     ("best", "Best (epoch 8)"),
                     ("latest", "Latest (epoch 30)"),
                     ("final", "Final")]:
    sd, meta = load_sd(name)
    epoch = meta.get('epoch', '?')
    best_r = meta.get('best_reward', '?')
    metrics = meta.get('metrics', {})
    print(f"\n{label} ({name}.pth):")
    print(f"  epoch={epoch}, best_reward={best_r}")
    if metrics:
        rw = metrics.get('avg_reward', '?')
        ls = metrics.get('rl_loss', '?')
        ac = metrics.get('accuracy', '?')
        kl = metrics.get('avg_kl', '?')
        rw_str = f"{rw:.4f}" if isinstance(rw, float) else str(rw)
        ls_str = f"{ls:.2f}" if isinstance(ls, float) else str(ls)
        ac_str = f"{ac:.1f}%" if isinstance(ac, (float,int)) else str(ac)
        kl_str = f"{kl:.4f}" if isinstance(kl, float) else str(kl)
        print(f"  metrics: reward={rw_str}  loss={ls_str}  acc={ac_str}  kl={kl_str}")
    print(f"  params={sum(p.numel() for p in sd.values()):,}")

# 2. 分析 tanh 位置编码参数（w, v, tau）
print("\n" + "="*60)
print("🔄 tanh 位置编码参数演化")
print("="*60)

for name, label in [("epoch0", "Epoch 0"), ("best", "Best"), ("final", "Final")]:
    sd, _ = load_sd(name)
    print(f"\n{label}:")
    for layer in range(4):
        w = sd[f'layers.{layer}.self_attn.w_param'].squeeze().tolist()
        v = sd[f'layers.{layer}.self_attn.v_param'].squeeze().tolist()
        log_tau = sd[f'layers.{layer}.self_attn.log_tau'].squeeze().tolist()
        tau = [np.exp(lt) + 0.1 for lt in log_tau]
        print(f"  层{layer}:")
        for h in range(4):
            print(f"    头{h}: w={w[h]:.4f}  v={v[h]:.4f}  τ={tau[h]:.4f}  (log_tau={log_tau[h]:.4f})")

# 3. Token Embedding 分析
print("\n" + "="*60)
print("🔤 Token Embedding 变化量")
print("="*60)

sd0, _ = load_sd("epoch0")
sdf, _ = load_sd("final")
emb0 = sd0['token_embedding.weight']
embf = sdf['token_embedding.weight']
diff = (embf - emb0).abs().mean(dim=1).tolist()

tokenizer = CharTokenizer()
# 尝试获取 id→char 映射
if hasattr(tokenizer, 'id_to_char'):
    id_to_char = tokenizer.id_to_char
elif hasattr(tokenizer, 'char_to_id'):
    id_to_char = {v: k for k, v in tokenizer.char_to_id.items()}
else:
    id_to_char = {}
    for i in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([i])
            id_to_char[i] = decoded if decoded else f'<{i}>'
        except:
            id_to_char[i] = f'<{i}>'
            
print("  每个 token 的 embedding 平均变化量（从 epoch0 到 final）：")
for i in range(tokenizer.vocab_size):
    char = id_to_char.get(i, f'<{i}>')
    if isinstance(char, bytes):
        char = char.decode('utf-8', errors='replace')
    print(f"    {str(char):>4} (id={i:2d}): Δ={diff[i]:.6f}")

print(f"\n  整体平均变化: {np.mean(diff):.6f}")
print(f"  最大变化: {np.max(diff):.6f} (token id={np.argmax(diff)})")

# 4. 权重统计（检查 NaN、死神经元、参数分布）
print("\n" + "="*60)
print("📊 权重分布统计")
print("="*60)

for name, label in [("epoch0", "Epoch 0"), ("final", "Final")]:
    sd, _ = load_sd(name)
    print(f"\n{label}:")
    all_vals = []
    nan_count = 0
    zero_count = 0
    total_count = 0
    for k, v in sd.items():
        vals = v.flatten()
        all_vals.append(vals)
        nan_count += torch.isnan(vals).sum().item()
        zero_count += (vals == 0).sum().item()
        total_count += vals.numel()
    all_vals = torch.cat(all_vals)
    print(f"  参数总量: {total_count:,}")
    print(f"  NaN 数量: {nan_count} {'⚠️' if nan_count > 0 else '✅'}")
    print(f"  精确为零: {zero_count} ({zero_count/total_count*100:.4f}%)")
    print(f"  均值: {all_vals.mean().item():.6f}")
    print(f"  标准差: {all_vals.std().item():.6f}")
    print(f"  最小值: {all_vals.min().item():.6f}")
    print(f"  最大值: {all_vals.max().item():.6f}")
    # 检查是否有梯度爆炸迹象
    p99 = torch.quantile(all_vals.abs(), 0.99).item()
    print(f"  绝对值 P99: {p99:.6f}")
    p999 = torch.quantile(all_vals.abs(), 0.999).item()
    print(f"  绝对值 P99.9: {p999:.6f}")

# 5. 逐层参数变化量（epoch0 → final）
print("\n" + "="*60)
print("📈 逐层参数变化量（epoch0 → final）")
print("="*60)

for layer in range(4):
    print(f"\n  层 {layer}:")
    for module in ['W_q', 'W_k', 'W_v', 'W_o']:
        w0 = sd0[f'layers.{layer}.self_attn.{module}.weight']
        wf = sdf[f'layers.{layer}.self_attn.{module}.weight']
        delta = (wf - w0).abs().mean().item()
        print(f"    {module}.weight: Δ={delta:.6f}")
    for module in ['ffn.0', 'ffn.3']:
        w0 = sd0[f'layers.{layer}.{module}.weight']
        wf = sdf[f'layers.{layer}.{module}.weight']
        delta = (wf - w0).abs().mean().item()
        print(f"    {module}.weight: Δ={delta:.6f}")

# output_proj
w0 = sd0['output_proj.weight']
wf = sdf['output_proj.weight']
print(f"\n  output_proj.weight: Δ={(wf-w0).abs().mean().item():.6f}")

# norm layers
for n in ['norm.weight', 'norm.bias']:
    if n in sd0 and n in sdf:
        delta = (sdf[n] - sd0[n]).abs().mean().item()
        print(f"  {n}: Δ={delta:.6f}")

