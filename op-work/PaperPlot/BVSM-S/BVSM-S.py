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
    
    #8:Ansor = [0.031, 0.024, 0.025, 0.024, 0.024, 0.032, 0.024, 0.024, 0.024, 0.024]   # BVSM-S:0.024 ,geo:0.025  AutoTVM = [0.024, 0.024, 0.024, 0.024, 0.024, 0.024, 0.024, 0.024, 0.024, 0.024]
    #errors_Ansor = [(0.001,0),(0.001,0),(0,0.001),(0.001,0),(0.001,0),(0,0.001),(0.001,0.001),(0.001,0),(0.001,0),(0.001,0)]
    #errors_AutoTVM = [(0.0002,0.00002),(0.00001,0.0002),(0.0001,0.0003),(0.0004,0),(0.0002,0),(0.0002,0.0001),(0.0001,0),(0.0001,0.0001),(0.0004,0),(0.0001,0.0001)]
    #16:Ansor = [0.11, 0.28, 0.11, 0.27, 0.11, 0.11, 0.23, 0.11, 0.11, 0.09]            # BVSM-S:0.094 ,geo:0.139  AutoTVM = [0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094]
    #errors_Ansor = [(0.008,0.001),(0,0.006),(0,0.001),(0.001,0.01),(0.003,0.002),(0.003,0),(0.005,0),(0.002,0),(0.008,0),(0.003,0)]
    #errors_AutoTVM = [(0.005,0.0006),(0.003,0),(0.004,0.0003),(0.0002,0),(0.002,0),(0.0005,0),(0.002,0),(0.002,0.001),(0.0003,0.002),(0.0002,0.001)]
    #24:Ansor = [0.40, 0.43, 0.50, 0.34, 0.30, 0.31, 0.30, 0.30, 0.29, 0.26]            # BVSM-S:0.285 ,geo:0.336  AutoTVM = [0.285, 0.285, 0.285, 0.285, 0.285, 0.285, 0.285, 0.285, 0.285, 0.285]
    #errors_Ansor = [(0.002,0.004),(0.008,0.003),(0.004,0.002),(0.006,0.001),(0,0.005),(0,0.001),(0.001,0.004),(0,0.004),(0.007,0.004),(0.001,0.003)]
    #errors_AutoTVM = [(0.003,0.003),(0.007,0.004),(0.003,0.001),(0.004,0.002),(0,0),(0.002,0.002),(0.001,0),(0.001,0.003),(0.002,0),(0.003,0.0008)]
    #32:Ansor = [0.704, 0.698, 0.905, 0.688, 0.708, 0.715, 0.708, 0.713, 0.692, 0.705]  # BVSM-S:0.663 ,geo:0.721  AutoTVM = [0.663, 0.663, 0.663, 0.663, 0.663, 0.663, 0.663, 0.663, 0.663, 0.663]
    #errors_Ansor = [(0.007,0.002),(0.006,0.003),(0.006,0.008),(0.007,0.005),(0.008,0.001),(0.009,0.002),(0.008,0.004),(0.005,0.005),(0.007,0.001),(0.001,0.004)]
    #errors_AutoTVM = [(0.003,0.002),(0.001,0.007),(0.004,0.002),(0.004,0.001),(0.0002,0.006),(0.002,0.0001),(0.0008,0.004),(0.002,0.0014),(0.0004,0),(0.002,0.003)]
    #40:Ansor = [1.13, 1.05, 1.03, 1.05, 1.07, 1.32, 1.07, 1.05, 1.04, 1.06]            # BVSM-S:1.022 ,geo:1.084  AutoTVM = [1.022, 1.022, 1.022, 1.022, 1.022, 1.022, 1.022, 1.022, 1.022, 1.022]
    #errors_Ansor = [(0.035,0.04),(0.029,0.03),(0.024,0.013),(0.054,0.018),(0.024,0.013),(0.045,0.012),(0.031,0.021),(0.043,0.028),(0.015,0.01),(0.014,0.032)]
    #errors_AutoTVM = [(0.004,0.001),(0.002,0.006),(0.001,0.003),(0.008,0),(0.002,0),(0.002,0.001),(0.006,0),(0.001,0.001),(0.004,0),(0.003,0.0008)]
    #48:Ansor = [1.27, 1.64, 1.22, 1.23, 1.25, 1.74, 1.22, 1.19, 1.15, 1.22]            # BVSM-S:1.256 ,geo:1.300  AutoTVM = [1.25, 1.26, 1.24, 1.24, 1.25, 1.25, 1.24, 1.26, 1.25, 1.26]
    #errors_Ansor = [(0.11,0.08),(0.07,0.07),(0.03,0.04),(0.03,0.06),(0.02,0.06),(0.06,0.07),(0.09,0.06),(0.03,0.05),(0.04,0.05),(0.07,0.05)]
    #errors_AutoTVM = [(0.008,0.04),(0.01,0.02),(0.02,0.03),(0.04,0.01),(0.02,0.03),(0.02,0.01),(0.01,0.01),(0.01,0.02),(0.01,0.03),(0.02,0.03)]
    #56:Ansor = [1.80, 1.46, 1.48, 1.51, 1.51, 1.5, 1.48,  1.51, 1.48, 1.47] AutoTVM = [1.50, 1.51, 1.49, 1.50, 1.50, 1.51, 1.50, 1.49, 1.50, 1.50]
    #errors_Ansor = [(0.12,0.1),(0.02,0.05),(0.06,0.09),(0.06,0.07),(0.09,0.04),(0.01,0.07),(0.01,0.05),(0.03,0.09),(0.02,0.08),(0.07,0.08)]
    #errors_AutoTVM = [(0.04,0.01),(0.01,0.02),(0.02,0.03),(0.04,0.03),(0.02,0.07),(0.05,0.02),(0.02,0.01),(0.01,0.03),(0.04,0.01),(0.01,0.03)]
    #64:
    Ansor = [2.01, 1.76, 1.67, 1.69, 1.78, 1.73, 2.03, 1.79, 1.76, 1.70]            # BVSM-S:1.622 ,geo:1.788  
    AutoTVM = [1.62, 1.61, 1.63, 1.63, 1.62, 1.61, 1.61, 1.61, 1.60, 1.63]   
    errors_Ansor = [(0.002,0.02),(0.07,0.11),(0.07,0.14),(0.08,0.14),(0.09,0.13),(0.06,0.13),(0.09,0.05),(0.07,0.08),(0.05,0.12),(0.07,0.12)]
    errors_AutoTVM = [(0.02,0.01),(0.01,0.03),(0.02,0.03),(0.04,0.01),(0.02,0.01),(0.02,0.01),(0.01,0.03),(0.04,0.06),(0.05,0.01),(0.01,0.04)]
    
    plt.figure(1)  # 创建第一个画板（figure）
    #plt.subplot(211)  # 第一个画板的第一个子图
    #fig, ax = plt.subplots()
    plt.ylabel('Time (us)', fontsize=20)
    font = FontProperties(fname=r"/home/daiwen/simhei/simhei.ttf",size=15)
    plt.xlabel('(64,64,64)', fontsize=24,fontproperties=font)
    
    plt.tick_params(labelsize=18)
    #plt.grid(linestyle=':', axis='y')
    x = np.arange(1,len(Ansor)+1)
    print(x)
    base = [1,1,1,1,1,1,1]
    
    l1 = plt.bar(x-0.1, Ansor, yerr=[[e[0] for e in errors_Ansor],[e[1] for e in errors_Ansor]], width=0.2, color='white', label='AnsorTVM', align='center', edgecolor='k', hatch='/')
    #l1 = plt.bar(x-0.1, Ansor, yerr=[[e[0] for e in errors_Ansor],[e[1] for e in errors_Ansor]], width=0.2, color='red', label='AnsorTVM', align='center')
    l2 = plt.bar(x+0.1, AutoTVM, yerr=[[e[0] for e in errors_AutoTVM],[e[1] for e in errors_AutoTVM]], width=0.2, color='white', label='AutoTVM', align='center', edgecolor='k', hatch='')
    #l2 = plt.bar(x+0.1, AutoTVM, yerr=[[e[0] for e in errors_AutoTVM],[e[1] for e in errors_AutoTVM]], width=0.2, color='blue', label='AutoTVM', align='center')
    plt.legend(labels=['AnsorTVM','BVSM-S'], loc='upper right', fontsize=20)
    plt.axhline(1.78, color='black',lw=2)

    plt.ylim(0.0, 2.2)
    #plt.legend(loc='upper right', fontsize=18)
    plt.savefig('BVSM-S-64.pdf',format='pdf', dpi=1000)
    #plt.show()

if __name__ == '__main__':
    Optimization_time_nor()
