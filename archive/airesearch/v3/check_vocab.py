import sys
sys.path.insert(0, "/root/airesearch/v2")
from v2 import CharTokenizer
t = CharTokenizer()
for i in range(t.vocab_size):
    dec = t.decode([i])
    print(f"{i}: {dec!r}")
