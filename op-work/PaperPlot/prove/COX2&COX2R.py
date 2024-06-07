# -*- coding:utf8 -*
# !/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#import brewer2mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#plt.rc('font',family='Times New Roman')

#plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体，不然中文无法显示
plt.rcParams['figure.figsize'] = (12.0, 6.0) # 设置figure_size尺寸
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
def geomean(ls):
    n = len(ls)
    r = 1
    for i in range(n):
       r *= ls[i]
    return pow(r,1/n)

def al_mean(ls):
    return np.average(ls)

def Optimization_time_nor():
    from matplotlib import ticker

    Ansor = [0.733,0.729]
    Atch = [0.0483,0.0486]
    Atch_padGroup = [0.0262,0.0263]
    
    fig, ax = plt.subplots()
    
    font = FontProperties(fname=r"/home/daiwen/simhei/simhei.ttf",size=15)
    plt.xlabel('RTX 3080上COX2与COX2_R不同实验的时间开销(ms)', fontsize=18,fontproperties=font)
    
    plt.tick_params(labelsize=18)
    plt.grid(linestyle=':', axis='y')
    y = np.arange(len(Ansor))
    base = [1,1,1,1,1,1,1]
    a1 = plt.barh(y-0.2, Ansor, 0.2, color='brown', label='Ansor', align='center')
    a2 = plt.barh(y, Atch, 0.2, color='yellow', label='BVSM-B', align='center')
    a3 = plt.barh(y+0.2, Atch_padGroup, 0.2, color='green', label='BVSM-G', align='center')
    
    y_tick = ['COX2', 'COX2_R']
    plt.yticks(y, y_tick,size=18,rotation=0)
    
    y = [-0.2,0.8,0,1,0.2,1.2]
    sum_data = Ansor + Atch + Atch_padGroup
    for i in range(len(y)):
        plt.text(0, y[i],sum_data[i], va = "center", ha = "left", size='18')
    
    plt.xlim(0.0, 0.1)
    plt.legend(loc='upper right', fontsize=18)
    plt.savefig('GPU_intel3_COX2与COX2_R.pdf',format='pdf', dpi=1000)
    plt.show()

if __name__ == '__main__':
    Optimization_time_nor()
