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
对于同构矩阵进行顺序处理
和批处理进行对比
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
        # attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )

    return [A, B, out]

# 获取矩阵并且逐一计算
def naive_GEMM(DS, M, K, N):

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

    print("len(matrices_A):",len(matrices_A))
    # 记录调优时间
    for i in range(0, len(matrices_A)):
        M = int(M)
        K = int(K)
        N = int(N)
        mytuner(M, K, N, i)

    #顺序计算时间的总和
    compute_time= sum(Time_1)
    print("%s的Ansor顺序计算的时间是: %f.3 (us)"%(DS,compute_time))



def mytuner(M, K, N, i):

    target = tvm.target.Target("llvm -mcpu=core-avx2")
    # target = tvm.target.Target("llvm -keys=cpu -link-params=0 -mcpu=core-avx2“）
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, K, N, "float32"), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)



    sch, args = task.apply_best("matmul_%d_%d_%d.json" % (M, K, N))
    log_file = "matmul_%d_%d_%d.json" % (M, K, N)
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
    np.testing.assert_allclose(C_np, C_tvm.numpy(), rtol=1e-3, atol = 1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, repeat=10, min_repeat_ms=500)
    rel = evaluator(A_tvm, B_tvm, C_tvm).results
    print(
        "Execution time of this operator: max:%.3f us   median:%.3f us   min:%.3f us"
        % (np.max(rel) * 1000000, np.median(rel) * 1000000, np.min(rel) * 1000000)
    )
    Time_1.append(np.median(rel)*1000000)
    #print("Equivalent python schedule:")
    #print(task.print_best(log_file))




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
    naive_GEMM(DS, M, K, N)
