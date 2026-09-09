import torch, sys
sys.path.insert(0, '/root/airesearch/baseline')
from baseline import StandardTransformer

# Load best model
model = StandardTransformer(16, 128, 4, 4, 512, 32)
ckpt = torch.load('/workspace/baseline/checkpoints/best.pth', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt)

# Get token embeddings
emb = model.token_embed.weight.detach()  # shape: [16, 128]

# Normalize for cosine similarity
emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
sim = emb_norm @ emb_norm.T  # [16, 16] cosine similarity matrix

# Token indices: 0-9 digits, 10='+', 11='-', 12='=', 13=PAD, 14=BOS, 15=EOS
names = ['0','1','2','3','4','5','6','7','8','9','+','-','=','PAD','BOS','EOS']

# 1. Average similarity between digits
digit_sim = 0.0
count = 0
for i in range(10):
    for j in range(i+1, 10):
        digit_sim += sim[i,j].item()
        count += 1
print(f'digits avg cos-sim: {digit_sim/count:.4f}')

# 2. Average similarity between digits and operators (+, -)
digit_op_sim = 0.0
count = 0
for i in range(10):
    for j in [10, 11]:  # + and -
        digit_op_sim += sim[i,j].item()
        count += 1
print(f'digit-op avg cos-sim: {digit_op_sim/count:.4f}')

# 3. Average similarity between operators
op_sim = sim[10,11].item()
print(f'+ vs - cos-sim: {op_sim:.4f}')

# 4. Find the top most similar pairs to '5'
print()
print('top similar to 5:')
sim_to_5 = [(names[i], sim[5,i].item()) for i in range(16) if i != 5]
sim_to_5.sort(key=lambda x: -x[1])
for name, s in sim_to_5[:8]:
    print(f'  {name}: {s:.4f}')

# 5. Check if numerical ordering is preserved
print()
adj_sim = sum(sim[i, i+1].item() for i in range(9))
far_sim = sum(sim[i, i+5].item() for i in range(5))
print(f'adjacent digits (3-4): {adj_sim/9:.4f}')
print(f'gap=5 (3-8):          {far_sim/5:.4f}')

# 6. Check digit-operator confusion: find if any digit is more similar to an op than to another digit
print()
print('worst confusions (digit-op sim > digit-digit sim):')
for i in range(10):
    min_digit_sim = min(sim[i, j].item() for j in range(10) if j != i)
    max_op_sim = max(sim[i, j].item() for j in [10, 11])
    if max_op_sim > min_digit_sim:
        print(f'  {names[i]}: nearest digit={min_digit_sim:.4f}, nearest op={max_op_sim:.4f} BAD!')
print('(if nothing printed, all digits are closer to other digits than to ops - good)')
