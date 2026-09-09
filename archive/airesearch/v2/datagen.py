# 生成一百以内加减法 CoT 数据
# 格式：a+b 或 a-b 的思维链（从右到左逐列计算）
# 返回：prompt + CoT 步骤 + 答案

import random

def gen_add_cot(a, b):
    """生成 a+b 的思维链"""
    a_str, b_str = str(a), str(b)
    n = max(len(a_str), len(b_str)) + 1
    a_pad, b_pad = a_str.zfill(n), b_str.zfill(n)

    lines = [f"{a}+{b}="]
    carry = 0

    for i in range(n - 1, -1, -1):
        da, db = int(a_pad[i]), int(b_pad[i])
        total = da + db + carry
        digit = total % 10
        carry = total // 10
        lines.append(f"{da}+{db}+{carry}→{digit}↑{carry}")

    lines.append(f"={a + b}")
    return "\n".join(lines)


def gen_sub_cot(a, b):
    """生成 a-b 的思维链（支持负结果）"""
    if a < b:
        inner = gen_sub_cot(b, a)
        inner_lines = inner.split("\n")
        result_val = -(b - a)
        cot = [f"{a}-{b}=", f"-({b}-{a})"]
        cot.extend(inner_lines[1:-1])
        cot.append(f"={result_val}")
        return "\n".join(cot)

    a_str, b_str = str(a), str(b)
    n = max(len(a_str), len(b_str))
    a_pad, b_pad = a_str.zfill(n), b_str.zfill(n)

    lines = [f"{a}-{b}="]
    borrow = 0

    for i in range(n - 1, -1, -1):
        da, db = int(a_pad[i]), int(b_pad[i])
        diff = da - db - borrow
        if diff >= 0:
            digit = diff
            borrow = 0
        else:
            digit = diff + 10
            borrow = 1
        lines.append(f"{da}-{db}-{borrow}→{digit}↑{borrow}")

    lines.append(f"={a - b}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("CoT 示例：")
    print(gen_add_cot(47, 58))
    print()
    print(gen_sub_cot(83, 47))
    print()
    print(gen_sub_cot(23, 47))
