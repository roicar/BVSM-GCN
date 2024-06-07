import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc('font', family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，使中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12.0, 10.0)  # 设置 figure_size 尺寸
plt.rcParams['savefig.dpi'] = 500  # 保存图片分辨率
plt.rcParams['figure.dpi'] = 500  # 分辨率

plt.style.use('fast')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rc('font', family='Times New Roman')

data_methods = ['AIDS', 'BZR', 'COX2', 'DHFR']
data_results = {
    'AIDS': [69, 68, 63.2, 83.2, 55.6, 55.8, 89.1, 76.4, 55.7, 85],
    'BZR': [42.5, 44.7, 38.5, 41.1, 28.4, 45, 43.8, 39.5, 41.7],
    'COX2': [64.3, 46.1, 45.4, 77.9, 65.0, 61.4, 66.0, 43.6, 46.6, 66.9],
    'DHFR': [62.8, 95.7, 102.0, 71.4, 100.7, 94.9, 88.1, 97.8, 78.9, 104.2]
}
BVSM = {
    'AIDS': [86.1, 85.9, 85.8, 85.8, 85.6],
    'BZR': [50.3, 50.2, 50, 49.9, 49.9],
    'COX2': [71.1, 71, 71, 70.9, 70.8],
    'DHFR': [121.1, 121, 121, 120.9, 120.8]
}

# 创建画布和子图
fig, ax = plt.subplots(figsize=(10, 6))
plt.ylabel('Time (us)', fontsize=18)
index = np.arange(len(data_methods))

# 绘制箱型图
box_data1 = [data_results[method] for method in data_methods]
box1 = ax.boxplot(box_data1, positions=index - 0.1, widths=0.2, manage_ticks=False, showfliers=False,
                  medianprops={'color': 'black'}, patch_artist=True, boxprops={'facecolor': 'white'})

box_data2 = [BVSM[method] for method in data_methods]
box2 = ax.boxplot(box_data2, positions=index + 0.1, widths=0.2, manage_ticks=False, showfliers=False,
                  medianprops={'color': 'black'}, patch_artist=True, boxprops={'facecolor': 'white'})

# 设置轴标签和图例
ax.set_xticks(index, data_methods, size=18)
ax.set_xticklabels(data_methods)

# 添加文本标注
for i in range(len(data_methods)):
    ax.text(index[i] - 0.1, max(data_results[data_methods[i]]) + 1, 'Intel-MKL-vbatch', ha='center', va='bottom')
    ax.text(index[i] + 0.1, max(BVSM[data_methods[i]]) + 1, 'BVSM', ha='center', va='bottom')

# 显示图形
plt.tight_layout()
plt.savefig('intel4_RealDS_mklbatch2.pdf', format='pdf', dpi=1000)
plt.show()
