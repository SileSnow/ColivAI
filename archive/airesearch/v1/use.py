import torch
import torch.nn as nn

model = torch.load("model.pth", weights_only=False)
model.eval()

print("🧮 简易计算器（输入 q 退出）")
print("格式：操作数1 运算符 操作数2")
print("运算符：0 = 加法，1 = 减法")
print("示例：23 0 45  →  23 + 45 =")
print("-" * 30)

while True:
    user_input = input("\n请输入：").strip()
    if user_input.lower() == "q":
        print("拜拜～👋")
        break
    
    try:
        a, op, b = map(int, user_input.split())
        if op not in [0, 1]:
            print("❌ 运算符只能是 0（加）或 1（减）")
            continue
        
        op_str = "+" if op == 0 else "-"
        
        X = torch.tensor([[a, op, b]], dtype=torch.float32)
        with torch.no_grad():
            output = model(X)
        
        print(f"  {a} {op_str} {b} = {output.item():.0f}")
    
    except ValueError:
        print("❌ 格式错误，请输入三个数字，空格隔开")
    except Exception as e:
        print(f"❌ 出错了：{e}")
