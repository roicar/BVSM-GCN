import numpy as np
import pandas as pd
import timeit
import sys
import os
import csv
import time
import torch

'''
这里pytorch库批处理矩阵乘法，作为一个baseline
'''


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

#获取数据集中各个相同维度矩阵乘法的数量
def get_P_B():
    file_dict = {}
    # 遍历文件夹中所有文件，将文件名前9位（对denseB而言）
    # 用字典来实现哈希表
    for file_name in file_names_B:
        key = file_name[:10]
        if key in file_dict:
            file_dict[key] += 1
        else:
            file_dict[key] = 1
    return file_dict

def get_P_A():
    file_dict = {}
    #遍历文件夹中所有文件，将文件名前10位（对sparseA而言）
    for file_name in file_names_A:
        key = file_name[:11]
        if key in file_dict:
            file_dict[key] += 1
        else:
            file_dict[key] = 1
    return file_dict



def pytorch_MM(DS, n_trials):
    # 用一个列表来获取A的矩阵
    for file_name_A in file_names_A:
        file_path = os.path.join(path, file_name_A)
        df = pd.read_csv(file_path, header=None)
        matrix = df.to_numpy()
        matrices_A.append(matrix)

    # 将返回的字典转化为列表
    file_dict_A = get_P_A()
    file_list_A = list(file_dict_A.items())

    # 用list_total_A 来接受分割开的矩阵列表
    list_total_A = []
    matrices_tmp_A = matrices_A
    for i in range(0, len(file_list_A)):
        tmp_p = file_list_A[i][1]
        list_tmp_A = list(matrices_tmp_A[:tmp_p])
        list_total_A.append(list_tmp_A)
        del matrices_tmp_A[:tmp_p]

    for file_name_B in file_names_B:
        file_path = os.path.join(path, file_name_B)
        df = pd.read_csv(file_path, header=None)
        matrix = df.to_numpy()
        matrices_B.append(matrix)

    file_dict_B = get_P_B()
    file_list_B = list(file_dict_B.items())

    # 用list_total_B 来接受分割开的矩阵列表
    list_total_B = []
    matrices_tmp_B = matrices_B
    for i in range(0, len(file_list_B)):
        tmp_p = file_list_B[i][1]
        list_tmp_B = list(matrices_tmp_B[:tmp_p])
        list_total_B.append(list_tmp_B)
        del matrices_tmp_B[:tmp_p]

    # print("len(file_list_B):", len(file_list_B))

    # 记录调优时间
    for i in range(0, len(file_list_B)):
        P = file_list_B[i][1]
        #print("P=", P)
        prefix = file_list_B[i][0]
        for file_name in file_names_B:
            if file_name.startswith(prefix):
                f_name = file_name

    torch_listA = []
    torch_listB = []


    for i in range(0, len(list_total_A)):
        torch_listA.append(torch.from_numpy(np.array(list_total_A[i])))
        torch_listB.append(torch.from_numpy(np.array(list_total_B[i])))

    # 存放n_trials次的执行时间
    EndTime = []
    # warm-up
    for i in range(0, len(list_total_A)):
        rel = torch.matmul(torch_listA[i], torch_listB[i])
    for i in range(0, int(n_trials)):
        # 记录开始时间
        start_time = time.time()
        for j in range(0, len(list_total_A)):
            rel = torch.matmul(torch_listA[j], torch_listB[j])
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
    # 计算的次数
    n_trials = str(sys.argv[2])
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
    pytorch_MM(DS, n_trials)





