import numpy as np
import pandas as pd
import timeit
import sys
import os
import csv
import time

'''
这个python文件用来获取各个数据集的图的不同节点数种数
也就是利用字典来获取种类个数
'''



def Get_dict(DS):

    #设置文件夹路径
    path = '../DataSets/'+DS+'/result/'

    #获取“sparseA”开头的所有矩阵
    file_names_A = [f for f in os.listdir(path) if f.startswith('sparseA')]

    file_dict = {}
    # 遍历文件夹中所有文件，将文件名前10位（对sparseA而言）
    for file_name in file_names_A:
        if file_name[9] == '_':
            key = file_name[8]
        else:
            key = file_name[8:10]
        if key in file_dict:
            file_dict[key] += 1
        else:
            file_dict[key] = 1
    return file_dict



    return

if __name__=='__main__':
    # DS 表示数据集名称
    DS = str(sys.argv[1])
    my_dict = Get_dict(DS)
    sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[0]))
    sum = 0
    for key,value in sorted_dict.items():
        print(key,value)
        val = int(value)
        key = int(key)
        sum = sum + 2*key*val*val*val
        print("computation: ",2*key*val*val*val)
        print("sum: ",sum)
        print("------------------------------")
