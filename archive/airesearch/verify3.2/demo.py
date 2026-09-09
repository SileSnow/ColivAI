"""verify3.2 demo — autoregressive generation examples"""
import torch, sys
sys.path.insert(0, '.')
from train import *

@torch.no_grad()
def generate(model, text, device):
    """Given e.g. '12+34=', autoregressively generate the answer."""
    ids = encode(text)
    eq_pos = ids.index(TOKEN2ID['='])
    prefix = ids[:eq_pos + 1]  # up to and including '='

    gen_ids = list(prefix)
    for _ in range(8):  # max answer length
        ids_tensor = torch.tensor([gen_ids], device=device)
        logits, _ = model(ids_tensor, None)
        next_token = logits[0, -1].argmax(dim=-1).item()
        gen_ids.append(next_token)
        if next_token == TOKEN2ID['EOS']:
            break

    # Decode
    result = ''.join(ID2TOKEN[t] for t in gen_ids)
    return result

def main():
    device = torch.device('cpu')

    # Load model
    model = MixedPETransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, D_FF).to(device)
    model.load_state_dict(torch.load('best.pth', map_location=device))
    model.eval()
    print(f"Loaded best model ({count_params(model):,} params)\n")

    # Test cases
    tests = [
        "12+34=", "56-23=", "99+1=", "0+0=", "7-15=",
        "45+55=", "83-99=", "10+20=", "77-33=", "3+8=",
        # Some edge cases
        "0-99=", "99-0=", "50+50=", "1-1=", "88+12=",
        "100-50=",  # wait, 100 is 3 digits... hmm
    ]

    # But our data only has 0-99 range, so a and b are max 2 digits
    tests = [
        "12+34=", "56-23=", "99+1=", "0+0=", "7-15=",
        "45+55=", "83-99=", "10+20=", "77-33=", "3+8=",
        "0-99=", "99-0=", "50+50=", "1-1=", "88+12=",
        "23+67=", "41-19=", "5+95=", "33-77=", "60+9=",
    ]

    correct = 0
    for t in tests:
        out = generate(model, t, device)
        # Parse: extract the answer after '='
        expected = t[-1] if t.endswith('=') else None
        # Actually compute expected
        expr = t[:-1]  # without '='
        if '+' in expr:
            a, b = expr.split('+')
            expected = str(int(a) + int(b))
        else:
            a, b = expr.split('-')
            expected = str(int(a) - int(b))

        # Extract model's answer
        # out format: "BOS...+...=...EOS"
        eq_pos = out.find('=')
        eos_pos = out.find('EOS')
        if eq_pos != -1:
            predicted = out[eq_pos+1:eos_pos] if eos_pos != -1 else out[eq_pos+1:]
        else:
            predicted = "?"

        ok = "✅" if predicted == expected else "❌"
        if predicted == expected:
            correct += 1
        print(f"  {t:<12} → {predicted:<6} (真: {expected:<6}) {ok}")

    print(f"\n  {correct}/{len(tests)} correct ({correct/len(tests):.1%})")

    # Show tanh params
    print(f"\n{'='*50}")
    print("tanh/sech² parameters per head:")
    for layer_idx in range(4):
        layer = model.layers[layer_idx]
        attn = layer.attn
        print(f"  Layer {layer_idx}:")
        for h in range(4):
            w = attn.w_param[h].item()
            v = attn.v_param[h].item()
            tau = attn.log_tau.exp().item()
            print(f"    Head {h}: w={w:+.4f}  v={v:+.4f}  τ={tau:.4f}")

if __name__ == '__main__':
    main()
