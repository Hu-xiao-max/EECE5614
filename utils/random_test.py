import random

def pick_a1_or_a2(a1, a2):
    # 随机生成 [0,1) 的浮点数，若小于 0.9 则返回 a1，否则返回 a2
    if random.random() < 0.9:
        return a1
    else:
        return a2

# 示例调用
a1 = 5
a2 = 99
counts = {a1: 0, a2: 0}

# 重复试验，统计输出频率
for _ in range(10000):
    res = pick_a1_or_a2(a1, a2)
    counts[res] += 1

print(counts)