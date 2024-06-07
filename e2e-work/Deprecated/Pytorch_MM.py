import torch
import time

# 生成随机的稠密矩阵
dense_matrix = torch.rand(100, 100)

# 生成随机的稀疏矩阵
import numpy as np
from scipy.sparse import random

# 定义矩阵的大小和稀疏度
matrix_size = (100, 100)
sparsity = 0.1  # 10% 稀疏度

# 生成稀疏矩阵
sparse_matrix = random(matrix_size[0], matrix_size[1], density=sparsity, format='coo')


# 将稠密矩阵转换为稀疏格式然后转换为 COO 格式
dense_matrix_sparse = dense_matrix.to_sparse().coalesce()

# 稠密矩阵乘法形式计算
start_time = time.time()
result_dense = torch.mm(dense_matrix, sparse_matrix.toarray())
end_time = time.time()
dense_time = end_time - start_time

# 稀疏矩阵乘法形式计算
start_time = time.time()
result_sparse = torch.sparse.mm(dense_matrix_sparse, sparse_matrix_coo)
end_time = time.time()
sparse_time = end_time - start_time

print("Dense format matrix multiplication time:", dense_time)
print("Sparse format matrix multiplication time:", sparse_time)
