import sys
import numpy as np
import pandas as pd
from scipy.sparse import random


if __name__ == '__main__':
    dim = int(sys.argv[1])
    No = str(sys.argv[2])
    # 生成一个128x128的稀疏矩阵，并保存为 CSV 文件
    rows, cols = dim, 5
    data = np.random.rand(rows, cols)
    
    # 生成随机数据
    df = pd.DataFrame(data)

    # 设置路径
    '''
    ../DataSets/R1/S1/result/
    
    PS:DenseC is random set(not equal to sparseA * denseB)
    We just test time,here no need to verify the MM.
    ''' 
    dim = str(dim)   
    path = "../DataSets/mkl_M32/result/"
    filename1 = "denseC_"+dim+"_"+dim+"_5_"+No+".csv"
    filename2 = "denseB_"+dim+"_"+dim+"_5_"+No+".csv"
    # 保存为 CSV 文件
    df.to_csv(path+filename1, index=False, header=False)
    df.to_csv(path+filename2, index=False, header=False)
    print(f"随机数据已保存到 {filename1}")
    print(f"随机数据已保存到 {filename2}")






