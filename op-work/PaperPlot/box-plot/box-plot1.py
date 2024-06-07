import matplotlib.pyplot as plt
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
# 生成一些随机数据作为示例
AIDS = [89.1,83.2,69,55.8,55.6]
BZR = [85.4,44.7,42.5,39.5,28.4]
COX2 = [77.9,66.0,64.3,46.1,43.6]
DHFR = [104,101,95.7,78.9,62.8]
D1 = [14.7,13.8,13.7,12.9,8.7]
D2 = [26.4,25.8,23.3,21.4,17.8]
D3 = [8.5,8.4,7.3,4.8,4.7]
D4 = [23.8,20.5,19.6,14.5,13.9]
D5 = [52.7,50.3,49.9,30.9,30.7]
data = [AIDS,BZR,COX2,DHFR,D1,D2,D3,D4,D5]
# 绘制箱型图
plt.boxplot(data, vert=True, patch_artist=False)


# 添加标签
plt.title('i9-11900K intel-MKL-vbatch')
x = np.arange(len(data))
base = [1,1,1,1,1,1,1,1,1]
plt.xticks(x+1, ['AIDS', 'BZR', 'COX2', 'DHFR', 'D1', 'D2', 'D3', 'D4', 'D5'])
#plt.xlabel('Datasets')
plt.ylabel('Times(us)')

# 显示图形
plt.savefig('intel3_CPU_BSMM_RandomSets.pdf',format='pdf', dpi=1000)
plt.show()
