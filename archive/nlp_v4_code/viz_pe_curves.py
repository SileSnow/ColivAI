"""
PE 消融 — 可视化: PE衰减曲线 + 学习参数
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

OUT = "/root/nlp_v4/figures"; os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════
# 图1: 四种PE衰减曲线理论对比
# ═══════════════════════════════════════════════
def plot_pe_theory():
    d = np.linspace(0, 200, 500)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # 左: 四种PE 偏置曲线对比
    ax = axes[0]
    ax.plot(d, -0.25 * d,                       'C0-',  lw=2.5, label='ALiBi (m=0.25)')
    ax.plot(d, 1.4*np.tanh(0.02*d)+1/np.cosh(np.clip(1.0*d,0,85))**2, 'C1--', lw=2, label='tanh + sech²')
    ax.plot(d, 1.4*np.arctan(0.02*d)+1/(1+(1.0*d)**2),          'C2-.', lw=2, label='arctan + 1/(1+d²)')
    # RoPE 等效: 余弦衰减包络 (多频复合的平均效果)
    rope_decay = np.mean([np.cos(0.01 * d / (2**i))**2 for i in range(6)], axis=0)
    ax.plot(d, rope_decay,                    'C3:',  lw=2.5, label='RoPE (multi-scale)')
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('Token Distance |i-j|', fontsize=12)
    ax.set_ylabel('Bias / Effective Weight', fontsize=12)
    ax.set_title('PE Attenuation vs Distance', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 200); ax.set_ylim(-5, 1.5)

    # 右: RoPE 多频率分解
    ax = axes[1]
    for fi, freq_idx in enumerate([0, 2, 4, 8, 16, 32, 48, 63]):
        theta = 1.0 / (10000 ** (2*freq_idx/64))
        decay = np.cos(theta * d)**2
        alpha = 0.3 + 0.6 * (fi/7)
        label = f'θ{freq_idx}' if fi%2==0 else None
        ax.plot(d, decay, color=plt.cm.viridis(fi/7), lw=1.5, alpha=alpha, label=label)
    ax.set_xlabel('Token Distance |i-j|', fontsize=12)
    ax.set_ylabel('Rotation Similarity', fontsize=12)
    ax.set_title('RoPE: Multi-Scale Frequency Decay (8 of 64 dim-pairs)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 200); ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(f"{OUT}/pe_theory_curves.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved: pe_theory_curves.png")

# ═══════════════════════════════════════════════
# 图2: tanh vs arctan 实际学习的参数对比
# ═══════════════════════════════════════════════
def plot_learned_params():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    
    for model_idx, (name, color, path) in enumerate([
        ('tanh+sech²', '#2196F3', '/root/nlp_v4/logs/mixed103_pe_params.json'),
        ('arctan+1/(1+d²)', '#FF9800', '/root/nlp_v4/logs/mixed_arctan_pe_params.json'),
    ]):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        with open(path) as f:
            params = json.load(f)
        
        layers = sorted(params.keys(), key=lambda x: int(x[1:]))
        nL = len(layers)
        
        # Collect all values per layer
        w_all, v_all, tau_all = [], [], []
        for l in layers:
            w_all.append(params[l]['w'])
            v_all.append(params[l]['v'])
            tau_all.append(params[l]['tau'])
        
        x = np.arange(nL)
        
        # w
        ax = axes[model_idx, 0]
        for li in range(nL):
            vals = w_all[li]
            ax.scatter([li]*4, vals, c=color, alpha=0.6, s=25)
            ax.plot([li-0.2, li+0.2], [np.mean(vals)]*2, c='red', lw=2)
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
        ax.set_title(f'w ({name})', fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=8)
        ax.set_ylabel('w value')
        
        # v
        ax = axes[model_idx, 1]
        for li in range(nL):
            vals = v_all[li]
            ax.scatter([li]*4, vals, c=color, alpha=0.6, s=25)
            ax.plot([li-0.2, li+0.2], [np.mean(vals)]*2, c='red', lw=2)
        ax.axhline(0.02, color='gray', ls='--', alpha=0.5, label='init')
        ax.axhline(0, color='red', ls=':', alpha=0.3)
        ax.set_title(f'v ({name})', fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=8)
        ax.set_ylabel('v value')
        
        # tau
        ax = axes[model_idx, 2]
        for li in range(nL):
            vals = tau_all[li]
            ax.scatter([li]*4, vals, c=color, alpha=0.6, s=25)
            ax.plot([li-0.2, li+0.2], [np.mean(vals)]*2, c='red', lw=2)
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
        ax.set_title(f'τ ({name})', fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=8)
        ax.set_ylabel('τ value')
    
    plt.suptitle('Learned PE Parameters: tanh vs arctan', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/learned_params.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved: learned_params.png")

# ═══════════════════════════════════════════════
# 图3: 最终 Val PPL 柱状图
# ═══════════════════════════════════════════════
def plot_final_ppl():
    models = ['sin/cos', 'ALiBi', 'tanh\n(WikiText-2)', 'RoPE', 'tanh\n(103)', 'tanh\n(LLaMA)', 'arctan']
    ppl    = [25.14, 24.73, 22.45, 10.31, 10.44, 10.69, 10.51]
    colors = ['#999', '#66B', '#F80', '#E31', '#F80', '#F80', '#F80']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, ppl, color=colors, edgecolor='white', linewidth=0.8)
    
    # 标注值
    for bar, val in zip(bars, ppl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Val PPL', fontsize=12)
    ax.set_title('Position Encoding Ablation Study — Final Results', fontsize=15)
    
    # 分组线
    ax.axvline(2.5, color='gray', ls='--', alpha=0.5)
    ax.text(1.25, 27, 'WikiText-2\n(BPE 2000)', ha='center', fontsize=9, color='gray')
    ax.text(4.5, 12, 'WikiText-103 (BPE 5000)', ha='center', fontsize=9, color='gray')
    
    ax.set_ylim(0, 28)
    plt.tight_layout()
    plt.savefig(f"{OUT}/final_ppl.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved: final_ppl.png")


if __name__ == '__main__':
    plot_pe_theory()
    plot_learned_params()
    plot_final_ppl()
    print(f"\n✅ All figures saved to {OUT}")
