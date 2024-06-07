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
'''
Ansor_naive.py
目的是用Ansor顺序计算某个数据集中的矩阵乘法
'''


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(M, K, N, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    out = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="out",
        
    )

    return [A, B, out]

# 获取矩阵并且逐一计算
def naive_GEMM(DS):

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

    for i in range(0, len(matrices_B)):
        print("shape_B的第%d个矩阵的K维度:" % (i), matrices_B[i].shape[0])
        print("shape_B的第%d个矩阵的N维度:" % (i), matrices_B[i].shape[1])

    # 这里设置临时调优时间用来存储相同GEMM的调优时间
    tuning_tmp_time = 0
    # 记录调优时间
    for i in range(0, len(matrices_A)):
        M = matrices_B[i].shape[0]
        K = matrices_B[i].shape[0]
        N = matrices_B[i].shape[1]
        tuning_tmp_time = mytuner(1000, M, K, N, i, 400, tuning_tmp_time)

    #顺序计算时间的总和
    compute_time= sum(Time_1)
    tune_time = sum(Time_2)
    print("%s的Ansor顺序计算的时间是: %f.3 (ms)"%(DS,compute_time))
    print("%s的Ansor顺序调优的时间是: %f.3 (s)" %(DS,tune_time))


def mytuner(n_trial, M, K, N, i, early_stopping, tuning_tmp_time):

    target = tvm.target.Target("llvm -mcpu=core-avx2")
    # target = tvm.target.Target("llvm -keys=cpu -link-params=0 -mcpu=core-avx2“）
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, K, N, "float32"), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    # 注意这里如果存在已经调优好的schedule
    # 那么可以直接用，不需要调优（调优时间直接累加）
    log_file = "matmul_%d_%d_%d.json" % (M, K, N)
    folder_path = "/home/daiwen/model_Batched_GEMM"
    file_list = glob.glob(os.path.join(folder_path, log_file))

    
    if len(file_list) > 0:
        Time_2.append(tuning_tmp_time)
        print("%s 的第 %d 个GEMM 的调优时间（已重复）为：%f.3 s" % (DS, i, tuning_tmp_time))
    else:
        
        tune_option = auto_scheduler.TuningOptions(
	    early_stopping=400,
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
        
        print("%s 的第 %d 个GEMM 的调优时间为：%f.3 s" % (DS, i, tuning_time))
        
    # Apply the best schedule
    sch, args = task.apply_best("matmul_%d_%d_%d.json" % (M, K, N))
    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))

    func = tvm.build(sch, args, target)
    A = matrices_A[i].astype(np.float32)
    B = matrices_B[i].astype(np.float32)
    C_np = np.matmul(A,B)

    dev = tvm.cpu()
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
    path = './DataSets/' + DS + '/result/'

    # 获取“denseB”和“sparseA”开头的所有矩阵
    file_names_A = [f for f in os.listdir(path) if f.startswith('sparseA')]
    file_names_B = [f for f in os.listdir(path) if f.startswith('denseB')]

    # 读取所有csv文件并转化为矩阵
    matrices_A = []
    matrices_B = []
    file_names_A.sort()
    file_names_B.sort()
    #表示计算时间
    Time_1 = []
    #表示调优时间
    Time_2 = []
    # mytuner(n_trial, DS, 400)
    naive_GEMM(DS)
