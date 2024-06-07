# -*- coding:utf8 -*
# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#import brewer2mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，不然中文无法显示
#plt.rc('font',family='Times New Roman')

#plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体，不然中文无法显示
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置figure_size尺寸
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
plt.rc('font', family='Times New Roman')

# 或者直接修改配色方案
#plt.rcParams['axes.color_cycle'] = colors
font_size = 18
def Data_power2_random_TEST():
    #plt.figure()
    plt.figure(1)  # 创建第一个画板（figure）
    plt.subplot(211)  # 第一个画板的第一个子图
    font = FontProperties(fname=r"/home/daiwen/simhei/simhei.ttf",size=15)
    plt.rc('font', family='Times New Roman')
    # plt.axhline(0.3197557 , color='red', linestyle='--')
    # plt.axvline(232, color='black', linestyle='--')
    # plt.axvline(240, color='green', linestyle='--')

    # plt.axhline(0.3197557 , color='red', linestyle='--')
    # plt.axvline(, color='black', linestyle='--')
    # plt.axvline(240, color='green', linestyle='--')
    plt.ylabel('Time(us)', fontsize=14)
    #plt.xlabel('Tasks', fontsize=14)

    y_Ansor = [17.427,12.849,11.443,37.412,19.126,21.664]
    y_order1 = [17.310,23.550,9.981,38.588,28.571,17.613]
    y_order2 = [20.889,15.669,11.466,48.144,19.612,20.875]
    y_order3 = [20.589,20.236,10.781,56.006,19.233,20.983]
    plt.tick_params(labelsize=14)
    plt.grid(linestyle='-.', axis='y')

    x = np.arange(len(y_Ansor))
    
    a1 = plt.bar(x-0.3, y_Ansor, 0.2, color='yellow', label='Ansor', align='center')
    a2 = plt.bar(x-0.1, y_order1, 0.2, color='red', label='order1', align='center')
    a3 = plt.bar(x+0.1, y_order2, 0.2, color='black', label='order2', align='center')
    a4 = plt.bar(x+0.3, y_order3, 0.2, color='blue', label='order3', align='center')
    
    plt.legend(labels=['Ansor','order1','order2','order3'], loc='upper left', fontsize=14)
    
    plt.xticks(x, ['(4,16,4)','(16,4,4)','(4,4,16)','(16,16,4)','(16,4,16)','(4,16,16)'],size=10,rotation=90)
    plt.ylim(0, 50)
    plt.savefig('循环序性能影响柱状图.pdf',format='pdf')

    #plt.show()

if __name__ == '__main__':
    Data_power2_random_TEST()
