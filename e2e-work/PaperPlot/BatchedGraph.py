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
 
    DGL_Sparse_batch = [12.82,3.7,4.69,7.45,17.12, 2.81, 4.73]
    BVSM_B = [0.127,0.068,0.046,0.084,0.013,0.012,0.0033]
    BVSM_M = [0.117,0.021,0.025,0.04,0.12,0.016,0.022]


    fig, ax = plt.subplots()
    ax.set_yscale('log')
    plt.ylabel('Time (ms)', fontsize=20)
    font = FontProperties(fname=r"/home/daiwen/simhei/simhei.ttf",size=15)

    plt.tick_params(labelsize=18)
    plt.grid(linestyle=':', axis='y')
    x = np.arange(len(BVSM_B))
    print(x)
    base = [1,1,1,1,1,1,1]

    
    #height = a0.get_height()
    #ax.text(a0.get_x() + a0.get_width()/2, height, f'{height:.0f}', ha = 'center', va='bottom')
    a1 = plt.bar(x-0.2, DGL_Sparse_batch, 0.2, color='#413E85', label='DGL*-Sparse', align='center')
    a2 = plt.bar(x, BVSM_B, 0.2, color='#30688D', label='TVM*-B', align='center')
    a3 = plt.bar(x+0.2, BVSM_M, 0.2, color='#35B777', label='TVM*-M', align='center')

    for a in [a1,a2,a3]:
        for bar in a:
            height = bar.get_height()
            ax.text(
		bar.get_x() + bar.get_width()/2, # x坐标
		height,
		f'{height}',
		ha = 'center',
		va = 'bottom'
		)

    plt.xticks(x, ['AIDS','BZR','COX2', 'DHFR','Letter-low','Cuneiform','Synthie'],size=18,rotation=0)
    plt.xticks(rotation=30)
    plt.ylim(0.0, 50)
    plt.legend(loc='upper right', fontsize=16)
    plt.savefig('EndtoEnd Inference of GCN with batched Graph.pdf',format='pdf', dpi=1000)
    #plt.show()

if __name__ == '__main__':
    Optimization_time_nor()
