import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rc('font', family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，以便中文能够显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12.0, 12.0)  # 设置 figure_size 尺寸
plt.rcParams['savefig.dpi'] = 500  # 保存图片分辨率
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.style.use('fast')
plt.figure()

def Data_power2_random_TEST():
    plt.figure(1)  # 创建第一个画板（figure）
    plt.subplot(211)  # 第一个画板的第一个子图
    plt.ylabel('Time (us)', fontsize=18)

    y_2 = [2.066, 3.002, 5.093, 13.091]
    y_3 = [1.59, 3.108, 5.796, 19.527]

    plt.tick_params(labelsize=14)
    x = np.arange(len(y_2))

    # 绘制散点图和折线图
    plt.scatter(x, y_2, color='black', linewidth='4')
    plt.scatter(x, y_3, color='black', linewidth='4')
    plt.plot(x, y_2, color="black", linewidth=3, linestyle='-')
    plt.plot(x, y_3, color="black", linewidth=3, linestyle='--')

    # 添加图例
    plt.legend(labels=['MKLvbatch', 'BVSM'], loc='upper left', fontsize=14)

    plt.xticks(x, ['4', '5', '6', '7'], size=14, rotation=0)
    annotation = 'logarithm of the number of floating-point operations'
    plt.xlabel(annotation, fontsize=18)

    plt.savefig("computation.pdf", format='pdf')

if __name__ == '__main__':
    Data_power2_random_TEST()
