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
'''
对某个数据集中特定维度的矩阵进行矩阵乘法

'''


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def batch_matmul(P, M, K, N, dtype):
    A = te.placeholder((P, M, K), name='A', dtype=dtype)
    B = te.placeholder((P, K, N), name='B', dtype=dtype)
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((P, M, N), lambda p, i, j: te.sum(A[p, i, k] * B[p, k, j], axis=k), name='C')

    return [A, B, C]




# 获取矩阵并且逐一计算
def batch_GEMM(DS, M, K, N):

    # 用一个列表来获取A的矩阵
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

    P = len(matrices_A)
    print("P = ", P)
    # 记录调优时间
    print("M,K,N=", M, K, N)
    tensorA = np.array(matrices_A)
    tensorB = np.array(matrices_B)


    M = int(M)
    K = int(K)
    N = int(N)

    mytuner(1000, P, M, K, N, tensorA, tensorB, 400)

    # 顺序计算时间的总和
    compute_time = sum(Time_1)
    tune_time = sum(Time_2)
    print("%s的Ansor批处理调优的时间是: %f (s)" % (DS, tune_time))


def mytuner(n_trial, P, M, K, N, tensorA, tensorB, early_stopping):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
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
    A = tensorA.astype(np.float32)
    B = tensorB.astype(np.float32)
    C_np = np.matmul(A, B)

    dev = tvm.cpu()
    A_tvm = tvm.nd.array(A, device=dev)
    B_tvm = tvm.nd.array(B, device=dev)

    C_tvm = tvm.nd.empty(C_np.shape, device=dev)

    func(A_tvm, B_tvm, C_tvm)

    # Check results
    np.testing.assert_allclose(C_np, C_tvm.numpy(), rtol=1e-3, atol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, repeat=10, min_repeat_ms=500)
    rel = evaluator(A_tvm, B_tvm, C_tvm).results

    print(
        "Execution time of this operator: max:%.3f us   median:%.3f us   min:%.3f us"
        % (np.max(rel) * 1000000, np.median(rel) * 1000000, np.min(rel) * 1000000)
    )


    print("Equivalent python schedule:")
    print(task.print_best(log_file))


if __name__ == '__main__':
    # DataSet
    DS = str(sys.argv[1])
    M = str(sys.argv[2])
    K = str(sys.argv[3])
    N = str(sys.argv[4])
    # 设置文件夹路径
    path = '../DataSets/' + DS + '/result/'

    # 获取“denseB”和“sparseA”开头的所有矩阵
    file_names_A = [f for f in os.listdir(path) if f.startswith('sparseA' + '_' + M + '_' + K + '_' + N)]
    file_names_B = [f for f in os.listdir(path) if f.startswith('denseB' + '_' + M + '_' + K + '_' + N)]

    # 读取所有csv文件并转化为矩阵
    matrices_A = []
    matrices_B = []
    file_names_A.sort()
    file_names_B.sort()
    # 表示计算时间
    Time_1 = []
    # 表示调优时间
    Time_2 = []
    batch_GEMM(DS, M, K, N)
