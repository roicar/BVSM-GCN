import logging
# testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
import time
import timeit
import math
import sys
import numpy as np
import tvm
from tvm import testing
from tvm import te, auto_scheduler
import os
import glob
import pandas as pd

'''BVSM.py
应用BVSM算法的起始程序
'''


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def batch_matmul(P, M, K, N, dtype):
    A = te.placeholder((P, M, K), name='A', dtype=dtype)
    B = te.placeholder((P, K, N), name='B', dtype=dtype)
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((P, M, N), lambda p, i, j: te.sum(A[p, i, k] * B[p, k, j], axis=k), name='C')

    return [A, B, C]


#After Data_retrive.py

#获取数据集中各个相同维度矩阵乘法的数量
def get_P_B():
    file_dict = {}
    # 遍历文件夹中所有文件，将文件名前9位（对denseB而言）
    # 用字典来实现哈希表
    for file_name in file_names_B:
        key = file_name[:9]
        if key in file_dict:
            file_dict[key] += 1
        else:
            file_dict[key] = 1
    return file_dict

def get_P_A():
    file_dict = {}
    #遍历文件夹中所有文件，将文件名前10位（对sparseA而言）
    for file_name in file_names_A:
        key = file_name[:10]
        if key in file_dict:
            file_dict[key] += 1
        else:
            file_dict[key] = 1
    return file_dict


def get_above_PadNumber(max):
    remainder = max%8
    if remainder==0:
        return max
    else:
        return max + (8 - remainder)

def get_below_PadNumber(min):
    remainder = min%8  
    if remainder==0:
        return min-7
    else:
        return min-remainder+1
    


# 获取一组矩阵中最大的M，K维度
def get_max_dimension():
    max_dims = [0,0] #获取矩阵的最大维度
    for matrix_A in matrices_A:
    	dims = matrix_A.shape
    	max_dims = [max(dims[i],max_dims[i]) for i in range(2)]
    	
    return max_dims

def get_min_dimension():
    min_dims = [2000,2000] #获取矩阵的最大维度
    for matrix_A in matrices_A:
    	dims = matrix_A.shape
    	min_dims = [min(dims[i],min_dims[i]) for i in range(2)]
    	
    return min_dims


# 返回分组后的矩阵        
def getGroup(DS,span,M_min,M_max):
    grouped_matrices_A = []
    grouped_matrices_B = []
    M_max_pad = get_above_PadNumber(M_max)
    M_min_pad = get_below_PadNumber(M_min)
    for i in range(M_min_pad, M_max_pad, span):
        group_A = [matrix for matrix in matrices_A if i <= matrix.shape[0] < i + span]
        grouped_matrices_A.append(group_A)
	
    for i in range(M_min_pad, M_max_pad, span):
        group_B = [matrix for matrix in matrices_B if i <= matrix.shape[0] < i + span]
        grouped_matrices_B.append(group_B)
	
    return grouped_matrices_A,grouped_matrices_B

# 获取矩阵并且逐一计算
def naive_GEMM(DS):
    max_dims = get_max_dimension()
    min_dims = get_min_dimension()
    M_min = min_dims[0]
    K_min = min_dims[1]
    M_max = max_dims[0]
    K_max = max_dims[1]
    Span = 8
    k=4
    t=1
    print("M_max=",M_max)
    print("M_min=",M_min)
    print("Span=",Span)
    Dmax = M_max
    Dmin = M_min
    N = matrices_B[0][1]
    
    P_total = len(matrices_A) # 用来记录矩阵总数
    
    if P_total == 1:
        print("BVSM is running BVSM_S strategy......")
        print("-------------------------------------")
        os.system('python BVSM_S.py '+ str(M_max) + ' ' + str(K_max) + ' ' + str(N))
    elif Dmax == Dmin:
        print("BVSM is running BVSM_B strategy......")
        print("-------------------------------------")
        os.system('python BVSM_B.py '+ str(DS))
    elif (Dmax-1)%Span==(Dmin-1)%Span: # 属于相同的section
        file_dict = get_P_B()
        max_key = max(file_dict, key=file_dict.get)
        min_key = min(file_dict, key=file_dict.get)
        Bmax = file_dict[max_key]
        Bmin = file_dict[min_key]
        if (Bmax/Bmin)<k:
            print("BVSM is running BVSM_M strategy......")
            print("-------------------------------------")
            os.system('python BVSM_M.py '+ str(DS))
        else:
            print("BVSM is running BVSM_B strategy......")
            print("-------------------------------------")
            os.system('python BVSM_B.py '+ str(DS))
    elif (Dmax-Dmin)<=t*Span:
        file_dict = get_P_B()
        max_key = max(file_dict, key=file_dict.get)
        min_key = min(file_dict, key=file_dict.get)
        Bmax = file_dict[max_key]
        Bmin = file_dict[min_key]
        if (Bmax/Bmin)<k:
            print("BVSM is running BVSM_M strategy......")
            print("-------------------------------------")
            os.system('python BVSM_M.py '+ str(DS))
        else:
            print("BVSM is running BVSM_B strategy......")
            print("-------------------------------------")
            os.system('python BVSM_B.py '+ str(DS))
    else:
        file_dict = get_P_B()
        max_key = max(file_dict, key=file_dict.get)
        min_key = min(file_dict, key=file_dict.get)
        Bmax = file_dict[max_key]
        Bmin = file_dict[min_key]
        if (Bmax/Bmin)<k:
            print("BVSM is running BVSM_G strategy......")
            print("-------------------------------------")
            os.system('python BVSM_G.py '+ str(DS))
        else:
            print("BVSM is running BVSM_B strategy......")
            print("-------------------------------------")
            os.system('python BVSM_B.py '+ str(DS))


if __name__ == '__main__':
    # DataSet
    DS = str(sys.argv[1])
    # 设置文件夹路径
    path = '../DataSets/' + DS + '/result/'

    # 获取“denseB”和“sparseA”开头的所有矩阵
    file_names_A = [f for f in os.listdir(path) if f.startswith('sparseA')]
    file_names_B = [f for f in os.listdir(path) if f.startswith('denseB')]

    # 读取所有csv文件并转化为矩阵
    matrices_A = []
    matrices_B = []
    file_names_A.sort()
    file_names_B.sort()
    # 表示计算时间
    MaxTime = []
    UpperTime = []
    MedianTime = []
    LowerTime = []
    MinTime = []
    # 表示调优时间
    Time_2 = []
	
    for file_name_A in file_names_A:
        file_path = os.path.join(path, file_name_A)
        df = pd.read_csv(file_path, header=None)
        matrix = df.to_numpy()
        matrices_A.append(matrix)
        
    for file_name_B in file_names_B:
        file_path = os.path.join(path, file_name_B)
        df = pd.read_csv(file_path, header=None)
        matrix = df.to_numpy()
        matrices_B.append(matrix)
        
    naive_GEMM(DS)













