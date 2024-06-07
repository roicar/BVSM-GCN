import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # 导入 Line2D 用于自定义图例

plt.rc('font', family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，以便中文能够显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12.0, 6.0)  # 设置 figure_size 尺寸
plt.rcParams['savefig.dpi'] = 500  # 保存图片分辨率
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.style.use('fast')
plt.figure()

def Data_power2_random_TEST():
    plt.figure(1)  # 创建第一个画板（figure）
    plt.subplot(211)  # 第一个画板的第一个子图
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    plt.ylabel('Time (ms)', fontsize=18)

    y_1 = [0.127, 0.127, 0.127, 0.127, 0.127]
    y_2 = [221, 67.4, 20.0, 12.82, 4.17]
    y_3 = [3.48, 1.70, 0.84, 0.117, 0.028]

    plt.tick_params(labelsize=14)
    x = np.arange(len(y_2))

    # 绘制散点图和折线图
    plt.scatter(x, y_1, color='red', linewidth='4')
    plt.scatter(x, y_2, color='blue', linewidth='4')
    plt.scatter(x, y_3, color='green', linewidth='4')
    line1, = plt.plot(x, y_1, color="red", linewidth=3, linestyle='-')
    line1, = plt.plot(x, y_2, color="blue", linewidth=3, linestyle='-')
    line2, = plt.plot(x, y_3, color="green", linewidth=3, linestyle='--')

    # 创建自定义图例
    legend_elements = [Line2D([0], [0], color='blue', lw=3, linestyle='-', label='DGL*-Sparse'),
                       Line2D([0], [0], color='red', lw=3, linestyle='-', label='TVM*-B'),
                       Line2D([0], [0], color='green', lw=3, linestyle='--', label='TVM*-M')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)

    plt.xticks(x, ['1','4', '16', '32', '128'], size=14, rotation=0)
    annotation = 'Batch size'
    plt.xlabel(annotation, fontsize=18)

    plt.savefig("batch_size.pdf", format='pdf')

if __name__ == '__main__':
    Data_power2_random_TEST()
