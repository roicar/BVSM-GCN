import numpy as np
import pandas as pd
import sys
from scipy.sparse import random


if __name__ == '__main__':
    dim = int(sys.argv[1])
    No = str(sys.argv[2])
    # 生成一个128x128的稀疏矩阵，并保存为 CSV 文件
    rows, cols = dim, dim
    #density = 0.01--0.1
    density = 0.1
    
    

    # 设置路径
    '''
    ../DataSets/R1/S1/result/
    '''
    
    dim = str(dim)
    
    path = "../DataSets/mkl_M32/result/"
    filename = "sparseA_"+dim+"_"+dim+"_5_"+No+".csv"
    # 生成稀疏矩阵
    sparse_matrix = random(rows, cols, density=density, format='csr')  # 使用 CSR 格式
    # 转换为 DataFrame
    df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
    
    # 保存为 CSV 文件
    df.to_csv(path+filename, index=False, header=False)

