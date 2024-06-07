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
import pandas as pd

'''Ansor_PadToMax.py
要求是自动对数据集中的矩阵进行pad和concatenate
这个pad到最大的方法适用于维度跨度较小的矩阵
比如D1，D2，D3数据集
保证数据集pad后最大的维度是4的倍数

注意下面的代码仅适用D1类的随机数据集（维度跨度小于8）
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


def getPadNumber(max):
    remainder = max%16
    if remainder==0:
        return max
    else:
        return max + (16 - remainder)
        
def get_above_PadNumber(max):
    remainder = max%8
    if remainder==0:
        return max
    else:
        return max + (8 - remainder)


# 获取一组矩阵中最大的M，K维度
def get_max_dimension():
    max_dims = [0,0] #获取矩阵的最大维度
    for matrix_A in matrices_A:
    	dims = matrix_A.shape
    	max_dims = [max(dims[i],max_dims[i]) for i in range(2)]
    	
    return max_dims

# 获取矩阵并且逐一计算
def naive_GEMM(DS):
    max_dims = get_max_dimension()
    M_max = max_dims[0]
    K_max = max_dims[1]
    print("M_max=",M_max)
    M_PadNumber = 0
    K_PadNumber = 0
    N = 0
    P = 0 # 用来记录矩阵总数
    M_PadNumber = get_above_PadNumber(M_max)
    K_PadNumber = M_PadNumber
    
    max_pad_A = [] # 初始化两个列表，存放pad完成后的数据集
    max_pad_B = []
    
    for matrix_A in matrices_A:
        M = matrix_A.shape[0]
        K = matrix_A.shape[1]
        matrix_A_pad = np.pad(matrix_A,((0,M_PadNumber - M),(0,K_PadNumber - K)),'constant',constant_values=0)
        max_pad_A.append(matrix_A_pad)
        P = P + 1

    for matrix_B in matrices_B:
        
        K = matrix_B.shape[0]  # K
        N = matrix_B.shape[1]  # N
        matrix_B_pad = np.pad(matrix_B, ((0, K_PadNumber - K), (0, 0)), 'constant', constant_values=0)
        max_pad_B.append(matrix_B_pad)

    print("M,K,N=",M_PadNumber,K_PadNumber,N)
    mytuner(1000, P, M_PadNumber, K_PadNumber, N, max_pad_A, max_pad_B, 400)
    
    compute_time = sum(Time_1)
    tune_time = sum(Time_2)
    print("%s的 PadToMax计算的时间是: %f (ms)" % (DS, compute_time))
    print("%s的 PadToMax调优的时间是: %f (s)" % (DS, tune_time))

def mytuner(n_trial, P, M, K, N, matrices_A, matrices_B, early_stopping):
    target = tvm.target.Target("cuda")
    # target = tvm.target.Target("llvm -keys=cpu -link-params=0 -mcpu=core-avx2“）
    task = tvm.auto_scheduler.SearchTask(func=batch_matmul, args=(P, M, K, N, "float32"), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "matmul_%d_%d_%d_%d.json" % (P, M, K, N)
    
    tune_option = auto_scheduler.TuningOptions(
        early_stopping=early_stopping,
        num_measure_trials=n_trial,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    # Run auto-tuning (search)
    start_time = time.time()
    task.tune(tune_option)
    end_time = time.time()
    tuning_time = end_time - start_time
    Time_2.append(tuning_time)
    
    # Apply the best schedule
    sch, args = task.apply_best("matmul_%d_%d_%d_%d.json" % (P, M, K, N))

    #print("Lowered TIR:")
    #print(tvm.lower(sch, args, simple_mode=True))

    func = tvm.build(sch, args, target)

    # Here dimension may have problems.
    A = np.array(matrices_A).astype(np.float32)
    B = np.array(matrices_B).astype(np.float32)
    C_np = np.matmul(A, B)

    dev = tvm.cuda()
    A_tvm = tvm.nd.array(A, device=dev)
    B_tvm = tvm.nd.array(B, device=dev)

    C_tvm = tvm.nd.empty(C_np.shape, device=dev)

    func(A_tvm, B_tvm, C_tvm)

    # Check results
    np.testing.assert_allclose(C_np, C_tvm.numpy(), rtol=1e-3)

    evaluator = func.time_evaluator(func.entry_name, dev, repeat=10, min_repeat_ms=500)
    rel = evaluator(A_tvm, B_tvm, C_tvm).results
    print("%s 的GEMM 的计算时间为：%f.3 ms"%(DS, np.median(rel) * 1000))
    Time_1.append(np.median(rel) * 1000)



    print("Equivalent python schedule:")
    print(task.print_best(log_file))



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
    Time_1 = []
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













