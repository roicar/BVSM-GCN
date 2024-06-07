import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
def plot_dimension_distribution(dimensions, values):

    # 绘制柱状图
    plt.rc('font', family='Times New Roman')
    #font = FontProperties(fname=r"/home/daiwen/simhei/simhei.ttf",size=20)
    fig, ax = plt.subplots(figsize=(12, 6))  # 创建一个大小为12x6的图形
    bars = ax.bar(dimensions, values)

    # 柱状图加入数值
    for bar in bars:
        height = bar .get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, # x坐标
            height,
            f'{height}',
            ha = 'center',
            va = 'bottom'
        )
    # 设置图形的标题和标签
    #ax.set_title('Matrix A distribution with various dimensions of M/K',fontsize = 20,fontproperties=font)
    ax.set_xlabel('M,K of Letter-low',fontsize = 18)
    ax.set_ylabel('Number of matrix A',fontsize = 18)

    # 设置x轴刻度和标签
    ax.set_xticks(dimensions)
    ax.set_xticklabels(dimensions)

    # 设置x轴标签的旋转角度,避免重叠
    plt.xticks(rotation=45)
    plt.ylim(0.0, 750)
    # 显示图形
    plt.tight_layout()
    plt.show()

# 示例用法

# COX2
#dimensions = [32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
#values = [2, 8, 11, 23, 47, 41, 29, 48, 57, 45, 39, 29, 25, 23, 12, 6, 4, 3, 4, 3, 4, 1, 1, 2]

# BZR
# dimensions = [13,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57]
# values = [2, 1, 1, 1, 2, 3, 1, 10, 3, 5, 3, 7, 5, 6, 27, 34, 26, 34, 27, 26, 26, 21, 14, 5,11 ,7,9,17,16,4,20,5,6,7,6,1,2,1,1,1,1]


# Cuneiform
# dimensions = [8, 16, 20, 24, 28, 32, 35, 36]
# values = [81, 55, 40, 55, 2, 1, 15, 18]


# Letter-low
dimensions = [1, 2, 3, 4, 5, 6, 7, 8]
values = [1, 128, 273, 551, 729, 407, 134, 27]


output_file = "Letter-low_Distribution.png"
plot_dimension_distribution(dimensions, values)

