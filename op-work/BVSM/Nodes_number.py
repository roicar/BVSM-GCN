import numpy as np
import pandas as pd
import timeit
import sys
import os
import csv
import time

'''Nodes_number.py
获取数据集中不同维度的矩阵种类数量
'''



def Get_Nodes_number(DS):

    #设置文件夹路径
    path = '../DataSets/'+DS+'/result/'

    #获取“sparseA”开头的所有矩阵
    file_names_A = [f for f in os.listdir(path) if f.startswith('sparseA')]

    #用集合存放文件名字前十位不同的名字（注意这里只考虑MK的不同）
    unique_names = set()
    print("%s数据集的MK不同的文件包括："%(DS))
    for file_name in file_names_A:
        prefix = file_name[:10]
        if prefix not in unique_names:
            unique_names.add(prefix)
            print(file_name)
    print("总共有%d个不同的Size:"%(len(unique_names)))
    unique_names_list = sorted(list(unique_names))

    #获取矩阵对应MK的值然后放入列表输出
    answer=[]
    for i in range(0,len(unique_names_list)):
        answer.append(unique_names_list[i][8] + unique_names_list[i][9])
    print(answer)



    return

if __name__=='__main__':
    # DS 表示数据集名称
    DS = str(sys.argv[1])
    Get_Nodes_number(DS)
