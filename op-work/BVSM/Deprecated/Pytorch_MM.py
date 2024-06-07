import numpy as np
import pandas as pd
import timeit
import sys
import os
import csv
import time
import torch

'''
测量pytorch的库对不同数据集的全部A*B=C的时间总和
默认库是intel-MKL
print(torch.__config__.show())
PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 10.2
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70
  - CuDNN 7.6.5
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=10.2, CUDNN_VERSION=7.6.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.12.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF,
'''


def pytorch_MM(DS, n_trials):
    # 设置文件夹路径
    path = './DataSets/' + DS + '/result/'

    # 获取“denseB”和“sparseA”开头的所有矩阵
    file_names_A = [f for f in os.listdir(path) if f.startswith('sparseA')]
    file_names_B = [f for f in os.listdir(path) if f.startswith('denseB')]

    # 读取所有csv文件并转化为矩阵
    matrices_A = []
    matrices_B = []
    file_names_A.sort()
    file_names_B.sort()
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

    '''
    matrix_B = np.loadtxt(open(path + 'denseB_' + Dim + '.csv'),delimiter=',',skiprows=0)
    matrix_A = np.loadtxt(open(path + 'sparseA_' + Dim + '.csv'),delimiter=',',skiprows=0)
    print("matrix_A.shape:",matrix_A.shape)
    print("matrix_B.shape:",matrix_B.shape)
    '''
    torch_listA = []
    torch_listB = []
    for i in range(0, len(matrices_A)):
        torch_listA.append(torch.from_numpy(matrices_A[i]))
        torch_listB.append(torch.from_numpy(matrices_B[i]))

    # 存放n_trials次的执行时间
    EndTime = []
    for i in range(0, int(n_trials)):
        # 记录开始时间
        start_time = time.time()
        # matmul方法计算乘积

        for i in range(0, len(matrices_A)):
            # print("i:",i)
            # print("matrices_A[i].shape:",matrices_A[i].shape)
            # print("matrices_B[i].shape:",matrices_B[i].shape)
            matrix_C = torch.matmul(torch_listA[i],torch_listB[i])
        # 记录结束时间
        end_time = time.time()
        # 计算执行时间
        exe_time = end_time - start_time
        EndTime.append(exe_time)
    max_time = max(EndTime)
    min_time = min(EndTime)
    n = len(EndTime)
    EndTime_sorted = sorted(EndTime)
    if n % 2 == 0:
        median_time = (EndTime_sorted[n // 2 - 1] + EndTime_sorted[n // 2]) / 2
    else:
        median_time = EndTime_sorted[n // 2]
    print("This is the execution time of %s:" % (DS))
    print("max:", max_time)
    print("min:", min_time)
    print("median:", median_time)
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


if __name__ == '__main__':
    # DS 表示数据集名称
    DS = str(sys.argv[1])
    # n_trials 表示最后测量执行时间episode的次数
    n_trials = str(sys.argv[2])
    pytorch_MM(DS, n_trials)





