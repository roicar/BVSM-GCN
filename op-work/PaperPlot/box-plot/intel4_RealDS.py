import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rc('font',family='Times New Roman')

plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，不然中文无法显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12.0, 10.0) # 设置figure_size尺寸
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
font_size = 24

data_methods = ['AIDS', 'BZR', 'COX2', 'DHFR']
data_results = {
    'AIDS': [69,68,63.2,83.2,55.6,55.8,89.1,76.4,55.7,85],
    'BZR': [42.5,44.7,38.5,41.1,28.4,45,43.8,39.5,85.4,41.7],
    'COX2': [64.3,46.1,45.4,77.9,65.0,61.4,66.0,43.6,46.6,66.9],
    'DHFR': [62.8,95.7,102.0,71.4,100.7,94.9,88.1,97.8,78.9,104.2]
}

BVSM = {
    'AIDS':[86.1,85.9,85.8,85.8,85.6],
    'BZR':[50.3,50.2,50,49.9,49.9],
    'COX2':[71.1,71,71,70.9,70.8],
    'DHFR':[121.1,121,121,120.9,120.8]
}
# 创建画布和子图
fig, ax = plt.subplots(figsize=(10, 6))
plt.ylabel('Time (us)', fontsize=18)
index = np.arange(len(data_methods))
# 绘制箱型图
box_data1 = [data_results[method] for method in data_methods]
ax.boxplot(box_data1, positions=index-0.1, widths=0.2, manage_ticks=False, showfliers=False, medianprops={'color':'white'},patch_artist = True, boxprops = {'facecolor':'grey'})
box_data2 = [BVSM[method] for method in data_methods]
ax.boxplot(box_data2, positions=index+0.1, widths=0.2, manage_ticks=False, showfliers=False, medianprops={'color':'white'})

# 设置轴标签和图例
ax.set_xticks(index,data_methods,size=18)
ax.set_xticklabels(data_methods)
#ax.set_xlabel('Methods',fontsize = 20)
#ax.set_ylabel('Values',fontsize = 20)

#plt.legend(loc='upper left', fontsize=18)

# 显示图形
#plt.tight_layout()
plt.savefig('intel4_RealDS_mklbatch2.pdf',format='pdf', dpi=1000)
#plt.show()
