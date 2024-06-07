import numpy as np

# 定义数据点 (x, y)
x = np.array([512, 4096, 13824, 32768, 64000, 110592])
y  = np.array([0.024, 0.094, 0.285, 0.663, 1.325, 1.646])
# 使用 numpy 的 polyfit 方法进行线性回归，得到斜率和截距
a, b = np.polyfit(x, y, 1)

print(f"斜率 a = {a}")
print(f"截距 b = {b}")

import matplotlib.pyplot as plt

# 使用之前计算的参数a和b来生成回归直线的y值
y_pred = a * x

# 绘制原始数据点
plt.scatter(x, y, color='blue', label='原始数据')

# 绘制回归直线
plt.plot(x, y_pred, color='red', label='回归直线')

# 添加图例
plt.legend()

# 设置x轴和y轴的标签
plt.xlabel('X')
plt.ylabel('Y')

# 显示图像
plt.show()
