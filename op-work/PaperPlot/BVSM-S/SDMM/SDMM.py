# -*- coding:utf8 -*
# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import brewer2mpl
import matplotlib.pyplot as plt
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
    # CONV = [0.4323,2.6223,0.2667,0.1940,0.1562,5.9893,0.3998,3.6136,1.9029,0.2530,0.4988,0.5417,0.0783,0.2424,0.2797,0.1857]
    # CONVOpt = [0.4312,2.6091,0.2649,0.2055,0.1542,5.9494,0.4627,3.7934,1.9969,0.2287,0.5379,0.5318,0.0737,0.2339,0.2656,0.1694]
    # Im2colCR = [0.8062,2.8291,0.6554,0.1638,0.1307,5.9892,0.2764,4.5594,2.2215,0.2290,0.7154,0.6816,0.0793,0.1256,0.1437,0.2626]
    # Im2colCD = [0.6808,2.5492,0.5817,0.1512,0.1074,5.8895,0.2553,4.3194,2.2165,0.5147,0.6418,0.6510,0.0780,0.1216,0.1392,0.2149]
    Ansor = [15.5,77.5,260,505,1092, 1234, 2006]
    SDMM = [15.4,55.2,210,462.8, 971,1136, 1669]
    # res = []
    # for i in range(len(AutoTVM)):
    #     res.append(AutoTVM[i]/AutoMCL[i])
    # print(res)

    # print("几何平均:",geomean(res))
    # print("算数平均:",al_mean(res))
    # res.append(geomean(res))



    fig, ax = plt.subplots()

    # plt.figure('Bar Chart', facecolor='lightgray')
    #plt.title('inception_v3', fontsize=18)
    #plt.xlabel('Schedule', fontsize=18)
    plt.ylabel('Time(us)', fontsize=18)
    #ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.tick_params(labelsize=18)
    plt.grid(linestyle=':', axis='y')
    x = np.arange(len(Ansor))
    print(x)
    base = [1,1,1,1,1,1,1]
    a = plt.bar(x-0.1, Ansor, 0.2, color='orange', label='Ansor', align='center')
    b = plt.bar(x+0.1, SDMM, 0.2, color='darkcyan', label='SDMM', align='center')
    #b = plt.bar(x + 0.2, Nor_CONVCD, 0.2, color='tomato', label='Im2colDNMM332', align='center')
    # 设置标签
    
    for i in a + b:
        h = i.get_height()
        plt.text(i.get_x() + i.get_width() / 2, h, '%2.1f'% (h), ha='center', va='bottom', size='10',rotation=90)
    
    #plt.axhline(1.0, color='red',lw=1,linestyle='--')
    plt.xticks(x, ['(8,8,8)','(16,16,16)','(24,24,24)','(32,32,32)','(40,40,40)','(48,48,48)','(64,64,64)'],size=8,rotation=90)
    plt.ylim(0.0, 2100)
    plt.legend(loc='upper left', fontsize=10)
    plt.savefig('SDMM.pdf',format='pdf', dpi=1000)
    #plt.show()

if __name__ == '__main__':
    Optimization_time_nor()
