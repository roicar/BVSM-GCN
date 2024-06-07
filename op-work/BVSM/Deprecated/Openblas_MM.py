import numpy as np
import pandas as pd
import timeit
import sys
import os
import csv
import time


'''
测量numpy的库对不同数据集的全部A*B=C的时间总和
这里用的numpy-1.23.0
默认库是openblas64_

numpy本身可以链接到不同的库
包括Intel MKL,ATLAS,BLIS

openblas64__info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None)]
    runtime_library_dirs = ['/usr/local/lib']
blas_ilp64_opt_info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None)]
    runtime_library_dirs = ['/usr/local/lib']
openblas64__lapack_info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None), ('HAVE_LAPACKE', None)]
    runtime_library_dirs = ['/usr/local/lib']
lapack_ilp64_opt_info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None), ('HAVE_LAPACKE', None)]
    runtime_library_dirs = ['/usr/local/lib']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2
    not found = AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL

'''

def numpy_MM(DS,n_trials):

    #设置文件夹路径
    path = '../DataSets/'+DS+'/result/'

    #获取“denseB”和“sparseA”开头的所有矩阵
    file_names_A = [f for f in os.listdir(path) if f.startswith('sparseA')]
    file_names_B = [f for f in os.listdir(path) if f.startswith('denseB')]

    #读取所有csv文件并转化为矩阵
    matrices_A = []
    matrices_B = []
    file_names_A.sort()
    file_names_B.sort()
    for file_name_A in file_names_A:
        file_path = os.path.join(path, file_name_A)
        df = pd.read_csv(file_path,header=None)
        matrix = df.to_numpy()
        matrices_A.append(matrix)

    for file_name_B in file_names_B:
        file_path = os.path.join(path, file_name_B)
        df = pd.read_csv(file_path,header=None)
        matrix = df.to_numpy()
        matrices_B.append(matrix)

    '''
    matrix_B = np.loadtxt(open(path + 'denseB_' + Dim + '.csv'),delimiter=',',skiprows=0)
    matrix_A = np.loadtxt(open(path + 'sparseA_' + Dim + '.csv'),delimiter=',',skiprows=0)
    print("matrix_A.shape:",matrix_A.shape)
    print("matrix_B.shape:",matrix_B.shape)
    '''

    # 存放n_trials次的执行时间
    EndTime = []
    for i in range(0, int(n_trials)):
        #记录开始时间
        start_time = time.time()
        #matmul方法计算乘积
        for i in range(0,len(matrices_A)):
            #print("i:",i)
            #print("matrices_A[i].shape:",matrices_A[i].shape)
            #print("matrices_B[i].shape:",matrices_B[i].shape)
            matrix_C = np.matmul(matrices_A[i],matrices_B[i])
        #记录结束时间
        end_time = time.time()
        #计算执行时间
        exe_time = end_time - start_time
        EndTime.append(exe_time)
    max_time = max(EndTime)
    min_time = min(EndTime)
    n = len(EndTime)
    EndTime_sorted = sorted(EndTime)
    if n % 2 == 0 :
        median_time = (EndTime_sorted[n//2 - 1] + EndTime_sorted[n//2])/2
    else:
        median_time = EndTime_sorted[n//2]
    print("This is the execution time of %s:"%(DS))
    print("max:",max_time)
    print("min:",min_time)
    print("median:",median_time)
    '''   
    #用dot方式计算乘积
    matrix_C2 = np.dot(matrix_A, matrix_B)
    print("This is result of matrix_A*matrix_B:\n", "matrix_C2.shape:", matrix_C2.shape)
    print("matrix_C2:\n", matrix_C2)

    
    由于matmul和dot在这里都是计算矩阵乘法
    且都有用到向量化
    matmul还是官方推荐的矩阵乘法函数
    所以后面计算时间只计算matmul作为numpy的基准
    '''

    '''
    #之前生成C结果的比对
    matrix_SC = np.loadtxt(open(path + 'denseC_4_4_4_25.csv'), delimiter=',', skiprows=0)
    print("This is a comparision:")
    print("matrix_SC.shape:",matrix_SC.shape)
    print("matrix_SC:\n",matrix_SC)

    #生成结果与numpy_MM.txt相加

    with open('./DataSets/'+DS+'/numpy_MM.txt','r') as f:
        rel = float(f.readline())
    print("rel:",rel)

    rel = rel + exe_time
    with open('./DataSets/'+DS+'/numpy_MM.txt','w') as f:
        f.write(str(rel))
    '''
    return


if __name__=='__main__':
    # DS 表示数据集名称
    DS = str(sys.argv[1])
    # n_trials 表示最后测量执行时间episode的次数
    n_trials = str(sys.argv[2])
    numpy_MM(DS, n_trials)





