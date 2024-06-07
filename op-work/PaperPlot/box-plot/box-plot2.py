import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rc('font',family='Times New Roman')

plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，不然中文无法显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (20.0, 6.0) # 设置figure_size尺寸
#plt.rcParams['figure.figsize'] = (4.0, 4.0) # 设置figure_size尺寸


#figsize(12.5, 4) # 设置 figsize
plt.rcParams['savefig.dpi'] = 500 #保存图片分辨率
plt.rcParams['figure.dpi'] = 500  #分辨率
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
# 指定dpi=200，图片尺寸为 1200*800
# 指定dpi=300，图片尺寸为 1800*1200
print(plt.style.available)
plt.style.use('fast')
#plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
#plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
# 参照下方配色方案，第三参数为颜色数量，这个例子的范围是3-12，每种配色方案参数范围不相同
#bmap = brewer2mpl.get_map('Set3', 'qualitative', 10)
#colors = bmap.mpl_colors
plt.figure()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
#plt.rc('font', family='Times New Roman')
plt.rc('font', family='Times New Roman')
# 或者直接修改配色方案
#plt.rcParams['axes.color_cycle'] = colors
font_size = 18

data_methods = ['AIDS', 'BZR', 'COX2']
data_results = {
    'AIDS': [89.1,83.2,69,55.8,55.6],
    'BZR': [42.5,44.7,38.5,41.1,28.4,45.0,43.8,39.5,85.4,41.7],
    'COX2': [77.9,66.0,64.3,46.1,43.6]
}

BVSM = {
    'AIDS': 50,
    'BZR': 60,
    'COX2': 70
}


# 创建画布和子图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bar_width = 0.3
index = np.arange(len(data_methods))


for i, method in enumerate(data_methods):
    plt.bar(index[i] - bar_width / 2, BVSM[method], bar_width, color='white', label=method, align='center', edgecolor='k', hatch='/')

# 绘制箱型图
box_data = [data_results[method] for method in data_methods]
ax.boxplot(box_data, positions=index + bar_width / 2, widths=bar_width, manage_ticks=False)

# 设置轴标签和图例
ax.set_xticks(index)
ax.set_xticklabels(data_methods)
ax.set_xlabel('Methods')
ax.set_ylabel('Values')

plt.legend()

# 显示图形
#plt.tight_layout()
plt.savefig('mklbatch.pdf',format='pdf', dpi=1000)
#plt.show()
