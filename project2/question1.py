import numpy as np
import matplotlib.pyplot as plt

# 示例矩阵（可替换为实际矩阵）
matrix = np.array([
    [11, 14, 12, 13],
    [13, 11, 14, 12],
    [12, 13, 11, 14],
    [14, 12, 13, 11]
])

# 映射箭头方向
arrow_map = {
    11: (0, 1),   # ↑
    12: (0, -1),  # ↓
    13: (-1, 0),  # ←
    14: (1, 0)    # →
}

# 创建绘图
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
ax.set_xticks(range(matrix.shape[1]))
ax.set_yticks(range(matrix.shape[0]))
ax.grid(True)

# 绘制箭头
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        dx, dy = arrow_map[matrix[i, j]]
        ax.arrow(j, matrix.shape[0] - 1 - i, dx * 0.3, -dy * 0.3, 
                 head_width=0.1, head_length=0.1, fc='black', ec='black')

# 显示图像
plt.show()