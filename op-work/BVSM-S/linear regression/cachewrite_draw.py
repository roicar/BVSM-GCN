import numpy as np

# 定义数据点 (x, y)
x = np.array([16, 24, 32, 40, 48, 56, 64])
y = np.array([-0.186, -0.133, -0.243, -0.188, -0.601, -0.342, -0.337])

# 使用 numpy 的 polyfit 方法进行线性回归，得到斜率和截距
a, b, c = np.polyfit(x, y, 2)

print(f" a = {a}")
print(f" b = {b}")
print(f" c = {c}")
import matplotlib.pyplot as plt

# 使用之前计算的参数a和b来生成回归直线的y值
y_pred = a * x*x + b*x + c

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
