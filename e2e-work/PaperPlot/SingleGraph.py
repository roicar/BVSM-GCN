# -*- coding:utf8 -*
# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#import brewer2mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#plt.rc('font',family='Times New Roman')

#plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体，不然中文无法显示
plt.rcParams['figure.figsize'] = (12.0, 8.0) # 设置figure_size尺寸
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
#plt.rcParams['font.sans-serif'] = ['Times New Roman']
#plt.rc('font', family='Times New Roman')
plt.rc('font', family='Times New Roman')
# 或者直接修改配色方案
#plt.rcParams['axes.color_cycle'] = colors
font_size = 18


def Optimization_time_nor():
    from matplotlib import ticker
 
    #DGL_Sparse = [221,45.5,56.16,89.34,254.2,32.04,57.89]
    #DGL_Dense = [171.1,36.9,42.7,70.6,191.2,24.3,44.7]
    #TVM_Sparse = [3.67,0.78,0.87,1.41, 4.29,0.47,0.89]
    #TVM_Dense = [3.48,0.72,0.79,1.28,3.9,0.452,0.76]

    DGL_Sparse = [221,45.5,56.16,89.34,254.2,32.04,57.89]
    DGL_Dense = [171.1,36.9,42.7,70.6,191.2,24.3,44.7]
    TVM_Sparse = [3.67,0.78,0.87,1.41, 4.29,0.47,0.89]
    TVM_Dense = [3.48,0.72,0.79,1.28,3.9,0.452,0.76]


    fig, ax = plt.subplots()
    ax.set_yscale('log')
    plt.ylabel('Time (ms)', fontsize=20)
    font = FontProperties(fname=r"/home/daiwen/simhei/simhei.ttf",size=15)
    #plt.xlabel('i9-11900K上四种随机数据集在BSMM不同优化方法下的时间开销', fontsize=18,fontproperties=font)
    #ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.tick_params(labelsize=18)
    plt.grid(linestyle=':', axis='y')
    x = np.arange(len(DGL_Sparse))
    print(x)
    base = [1,1,1,1,1,1,1]

    a0 = plt.bar(x-0.3, DGL_Sparse, 0.2, color='#413E85', label='DGL*-Sparse', align='center')
    #height = a0.get_height()
    #ax.text(a0.get_x() + a0.get_width()/2, height, f'{height:.0f}', ha = 'center', va='bottom')
    a1 = plt.bar(x-0.1, DGL_Dense, 0.2, color='#30688D', label='DGL*-Dense', align='center')
    a2 = plt.bar(x+0.1, TVM_Sparse, 0.2, color='#35B777', label='TVM', align='center')
    a3 = plt.bar(x+0.3, TVM_Dense, 0.2, color='#F8E620', label='TVM*-Dense', align='center')
    
    for a in [a0,a1,a2,a3]:
        for bar in a:
            height = bar.get_height()
            ax.text(
		bar.get_x() + bar.get_width()/2, # x坐标
		height,
		f'{height}',
		ha = 'center',
		va = 'bottom'
		)
    
    #plt.axhline(1.0, color='red',lw=1,linestyle='--')
    plt.xticks(x, ['AIDS','BZR','COX2','DHFR','Letter-low','Cuneiform','Synthie'],size=18,rotation=0)
    plt.xticks(rotation=30)
    plt.ylim(0.0, 600)
    plt.legend(loc='upper right', fontsize=16)
    plt.savefig('EndtoEnd Inference of GCN with Single Graph.pdf',format='pdf', dpi=1000)
    #plt.show()

if __name__ == '__main__':
    Optimization_time_nor()
