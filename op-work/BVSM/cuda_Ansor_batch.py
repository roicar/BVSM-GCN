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
import fnmatch
import glob
'''
cuda
用Ansor在GPU上顺序进行矩阵乘法作为一个naive的基准
实验所用数据集为COX2，
比较对象是串行cusparse
'''


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(P, M, K, N, dtype):
    A = te.placeholder((P, M, K), name='A', dtype=dtype)
    B = te.placeholder((P, K, N), name='B', dtype=dtype)
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((P, M, N), lambda p, i, j: te.sum(A[p, i, k] * B[p, k, j], axis=k), name='C')

    return [A, B, C]

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

# 获取矩阵并且逐一计算
def batch_GEMM(DS):
    print("fileA_number:",len(file_names_A))
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

    print("file_list_B_len:",len(file_list_B))
    # 这里设置临时调优时间用来存储相同GEMM的调优时间
    tuning_tmp_time = 0
    # 记录调优时间
    for i in range(0, len(file_list_B)):
        P = file_list_B[i][1]
        print("P=", P)
        prefix = file_list_B[i][0]
        for file_name in file_names_B:
            if file_name.startswith(prefix):
                f_name = file_name

        file_path = os.path.join(path, f_name)
        df = pd.read_csv(file_path, header=None)
        matrix = df.to_numpy()
        M = matrix.shape[0]
        K = matrix.shape[0]
        N = matrix.shape[1]
        print("P,M,K,N",P,M,K,N)
        tuning_tmp_time = mytuner(1000, P, M, K, N, i, list_total_B,list_total_A,400)

    #顺序计算时间的总和
    compute_time= sum(Time_1)
    tune_time = sum(Time_2)
    print("%s的Ansor批处理计算的时间是: %f.3 (ms)"%(DS,compute_time))
    print("%s的Ansor批处理调优的时间是: %f.3 (s)" %(DS,tune_time))


def mytuner(n_trial, P, M, K, N, i,list_total_B, list_total_A,early_stopping):

    target = tvm.target.Target("cuda")
    # target = tvm.target.Target("llvm -keys=cpu -link-params=0 -mcpu=core-avx2“）
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(P, M, K, N, "float32"), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    # 注意这里如果存在已经调优好的schedule
    # 那么可以直接用，不需要调优（调优时间直接累加）
    log_file = "matmul_%d_%d_%d_%d.json" % (P, M, K, N)
    folder_path = "/home/daiwen/model_Batched_GEMM/"
    file_list = glob.glob(os.path.join(folder_path, log_file))



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
    tuning_tmp_time = tuning_time
    print("%s 的第 %d 个维度GEMM 的调优时间为：%f.3 s" % (DS, i, tuning_time))
    # Apply the best schedule
    sch, args = task.apply_best("matmul_%d_%d_%d_%d.json" % (P, M, K, N))

    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))

    func = tvm.build(sch, args, target)
    # 随机
    # A = np.random.uniform(size=(P, M, K)).astype(np.float32)
    # B = np.random.uniform(size=(P, K, N)).astype(np.float32)
    # C_np = np.matmul(A,B)

    A = np.array(list_total_A[i]).astype(np.float32)
    B = np.array(list_total_B[i]).astype(np.float32)


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
    print("%s 的第%d个GEMM 的计算时间为：%f.3 ms" % (DS, i, np.median(rel) * 1000))
    Time_1.append(np.median(rel)*1000)
    # print("Equivalent python schedule:")
    # print(task.print_best(log_file))

    return tuning_tmp_time


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
    # mytuner(n_trial, DS, 400)
    batch_GEMM(DS)
