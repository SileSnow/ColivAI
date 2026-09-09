import sys
sys.path.insert(0, "/root/airesearch/v2")
from v2 import CharTokenizer

t = CharTokenizer()
print(f"vocab_size: {t.vocab_size}")

enc = t.encode("12+34=")
print(f"encode('12+34='): {enc}")

encs = t.encode("12+34=", add_special=True)
print(f"encode('12+34=', add_special=True): {encs}")

dec = t.decode([1, 2, 3])
print(f"decode([1,2,3]): {dec!r}")

# list all chars
print("\nAll tokens:")
if hasattr(t, 'char_to_id'):
    for c, i in sorted(t.char_to_id.items(), key=lambda x: x[1]):
        print(f"  id={i:2d}: {c!r}")
elif hasattr(t, 'id_to_char'):
    for i, c in sorted(t.id_to_char.items()):
        print(f"  id={i:2d}: {c!r}")
else:
    print(f"  tokenizer type: {type(t)}")
    print(f"  dir: {[x for x in dir(t) if not x.startswith('_')]}")
