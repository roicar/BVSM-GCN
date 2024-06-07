import numpy as np
import pandas as pd
import random

def generate_matrix_dataset(M, K, N, quantity, path):
    matrices = []
    # quantity表示这个维度的矩阵的数量
    for i in range(quantity):
        matrix_A = np.random.rand(M, K).astype(float)
        df_a = pd.DataFrame(matrix_A)
        df_a.to_csv(path+"sparseA_"+str(M)+"_"+str(K)+"_"+str(N)+"_"+str(i)+".csv", index=False, header=False)

        matrix_B = np.random.rand(K, N).astype(float)
        df_b = pd.DataFrame(matrix_B)
        df_b.to_csv(path+"denseB_"+str(M)+"_"+str(K)+"_"+str(N)+"_"+str(i)+".csv", index=False, header=False)

        matrix_C = np.matmul(matrix_A, matrix_B)
        df_c = pd.DataFrame(matrix_C)
        df_c.to_csv(path+"denseC_"+str(M)+"_"+str(K)+"_"+str(N)+"_"+str(i)+".csv", index=False, header=False)



    print(str(M) + "_" + str(K) + "_" + str(N) + "数据集已生成并保存。")

if __name__ == '__main__':
    # 这里从简假设M=K
    # 因为受控参数过多，就不从键盘输入了
    # 这里假设N=[2,8]
    N = random.randint(2,8)

    # 设置路径，三个数据集
    path = "DataSets/D5/result/"

    # 变维小矩阵M范围
    # D1：随机生成7个范围在32---40之间的数代表M
    # D2: 40---48
    # D3: 14个1---16
    # D4: 17---32
    # 注意这里要求维度不会重复（因为重复的维度在下一个层次）
    randomM = random.sample(range(33,49),14)
    for i in range(0,14):
        M = randomM[i]
        K = M
        quantity = random.randint(1,50)
        generate_matrix_dataset(M, K, N, quantity, path)
