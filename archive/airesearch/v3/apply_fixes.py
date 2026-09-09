with open("/root/airesearch/v3/v3_grpo_chain.py", "r") as f:
    code = f.read()

changes = 0

old = "RL_KL_BETA      = 0.01   # GRPO 无偏 KL 惩罚系数"
new = "RL_KL_BETA      = 0.05   # GRPO 无偏 KL 惩罚系数（调高防止 KL 爆表）"
if old in code:
    code = code.replace(old, new)
    changes += 1
    print("RL_KL_BETA: 0.01 -> 0.05")

old = "RL_LR       = 1e-4       # RL 学习率"
new = "RL_LR       = 3e-5       # RL 学习率（降低防止一步跨太远）"
if old in code:
    code = code.replace(old, new)
    changes += 1
    print("RL_LR: 1e-4 -> 3e-5")

old = "                d_clamped = torch.clamp(d, min=-20, max=20)"
new = "                d_clamped = torch.clamp(d, min=-20, max=20)\n                kl_per_token = torch.clamp(kl_per_token, max=5.0)"
# Need to find the exact block
old_block = """                d_clamped = torch.clamp(d, min=-20, max=20)
                kl_per_token = torch.exp(d_clamped) - d - 1
                if torch.isnan(kl_per_token).any():"""

new_block = """                d_clamped = torch.clamp(d, min=-20, max=20)
                kl_per_token = torch.exp(d_clamped) - d - 1
                kl_per_token = torch.clamp(kl_per_token, max=5.0)
                if torch.isnan(kl_per_token).any():"""

if old_block in code:
    code = code.replace(old_block, new_block)
    changes += 1
    print("per-token KL clamp: max=5.0")
else:
    print("WARNING: KL clamp block not found!")

print(f"\nTotal changes: {changes}")

with open("/root/airesearch/v3/v3_grpo_chain.py", "w") as f:
    f.write(code)

import py_compile
try:
    py_compile.compile("/root/airesearch/v3/v3_grpo_chain.py", doraise=True)
    print("Syntax check PASSED")
except py_compile.PyCompileError as e:
    print(f"Syntax ERROR: {e}")
