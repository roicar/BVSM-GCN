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

'''Ansor_PadGroup.py
Ansor_padGroup是自动对一个较大跨度的数据集先进行分组，然后对每组进行Pad
这里适用于D1，D2
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
    print("M_max=",M_max)
    print("M_min=",M_min)
    M_PadNumber = 0
    K_PadNumber = 0
    span=8
    N = 0
    P_total = len(matrices_A) # 用来记录矩阵总数
    
    grouped_matrices_A,grouped_matrices_B = getGroup(DS,span,M_min,M_max)
    
    group_pad_A = []
    group_pad_B = []
    P_list = []
    
    print("len=",len(grouped_matrices_A))
    for i in range(len(grouped_matrices_A)):
    	tmp_pad_A = [] # 清空并且临时存放一组中pad完的矩阵
    	tmp_pad_B = []
    	tmp_P = 0
    	for matrix_A in grouped_matrices_A[i]:
    	    #print("-------------------------------")
    	    M = matrix_A.shape[0]
    	    #print("M=",M)
    	    K = matrix_A.shape[1]
    	    #print("K=",K)
    	    M_PadNumber = get_above_PadNumber(M)
    	    #print("M_PadNumber=",M_PadNumber)
    	    K_PadNumber = M_PadNumber
    	    #print("K_PadNumber=",K_PadNumber)
    	    matrix_A_pad = np.pad(matrix_A,((0,M_PadNumber - M),(0,K_PadNumber - K)),'constant',constant_values=0)
    	    tmp_pad_A.append(matrix_A_pad)
    	    tmp_P = tmp_P + 1
    	
    	matrix_padA = np.array(tmp_pad_A)
    	#print("shape=",matrix_padA.shape)
    	group_pad_A.append(matrix_padA) # 把这一个小组的矩阵保存起来
    	
    	for matrix_B in grouped_matrices_B[i]:
    	    K = matrix_B.shape[0]
    	    N = matrix_B.shape[1]
    	    K_PadNumber = get_above_PadNumber(K)
    	    matrix_B_pad = np.pad(matrix_B,((0,K_PadNumber - K),(0,0)),'constant',constant_values=0)
    	    tmp_pad_B.append(matrix_B_pad)
    	
    	matrix_padB = np.array(tmp_pad_B)
    	group_pad_B.append(tmp_pad_B)
    	P_list.append(tmp_P)
    
    for i in range(len(group_pad_A)):
    	M = group_pad_A[i].shape[1]
    	print("M=",M)
    	K = M
    	P = P_list[i]
    	print("P=",P)    	
    	mytuner(1000, P, M, K, N, group_pad_A[i], group_pad_B[i], 400)
    
    # 顺序计算时间的总和
    Maxtime = sum(MaxTime)
    Uppertime = sum(UpperTime)
    Mediantime = sum(MedianTime)
    Lowertime = sum(LowerTime)
    Mintime = sum(MinTime)
    print("%s的Ansor PadGroup计算时间最大值是: %f (ms)" % (DS, Maxtime))
    print("%s的Ansor PadGroup计算时间75是: %f (ms)" % (DS, Uppertime))
    print("%s的Ansor PadGroup计算时间中间值是: %f (ms)" % (DS, Mediantime))
    print("%s的Ansor PadGroup计算时间25是: %f (ms)" % (DS, Lowertime))
    print("%s的Ansor PadGroup计算时间最小值是: %f (ms)" % (DS, Mintime))
    #print("%s的Ansor PadGroup调优的时间是: %f (s)" % (DS, tune_time))
    
    
    
 
    
    

    

def mytuner(n_trial, P, M, K, N, matrices_A, matrices_B, early_stopping):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    # target = tvm.target.Target("llvm -keys=cpu -link-params=0 -mcpu=core-avx2“）
    task = tvm.auto_scheduler.SearchTask(func=batch_matmul, args=(P, M, K, N, "float32"), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "matmul_%d_%d_%d_%d.json" % (P, M, K, N)
    folder_path = "/home/daiwen/model_Batched_GEMM"
    file_list = glob.glob(os.path.join(folder_path, log_file))

    
    if len(file_list) > 0:
        print("PASS")
    else:
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
    
    print("A.shape[0]:",A.shape[0])
    print("A.shape[1]:",A.shape[1])
    print("A.shape[2]:",A.shape[2])
    print("B.shape[0]:",B.shape[0])
    print("B.shape[1]:",B.shape[1])
    print("B.shape[2]:",B.shape[2])
    dev = tvm.cpu()
    A_tvm = tvm.nd.array(A, device=dev)
    B_tvm = tvm.nd.array(B, device=dev)

    C_tvm = tvm.nd.empty(C_np.shape, device=dev)

    print("A_tvm.shape[0]:",A_tvm.shape[0])
    print("A_tvm.shape[1]:",A_tvm.shape[1])
    print("A_tvm.shape[2]:",A_tvm.shape[2])
    print("B_tvm.shape[0]:",B_tvm.shape[0])
    print("B_tvm.shape[1]:",B_tvm.shape[1])
    print("B_tvm.shape[2]:",B_tvm.shape[2])
    func(A_tvm, B_tvm, C_tvm)

    # Check results
    np.testing.assert_allclose(C_np, C_tvm.numpy(), rtol=1e-3)

    evaluator = func.time_evaluator(func.entry_name, dev, repeat=10, min_repeat_ms=500)
    rel = evaluator(A_tvm, B_tvm, C_tvm).results
    np.sort(rel)
    print("%s 的GEMM 的计算时间最大值为：%f.3 ms"%(DS, np.max(rel) * 1000))
    print("%s 的GEMM 的计算时间75的数为"%(DS),np.percentile(rel, 75)*1000)
    print("%s 的GEMM 的计算时间中位数为：%f.3 ms"%(DS, np.median(rel) * 1000))
    print("%s 的GEMM 的计算时间25的数为"%(DS),np.percentile(rel, 25)*1000)
    print("%s 的GEMM 的计算时间最小值为：%f.3 ms"%(DS, np.min(rel) * 1000))
    MaxTime.append(np.max(rel) * 1000)
    UpperTime.append(np.percentile(rel, 75)*1000)
    MedianTime.append(np.median(rel) * 1000)
    LowerTime.append(np.percentile(rel, 25)*1000)
    MinTime.append(np.min(rel) * 1000)


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













