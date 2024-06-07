# joblib是一组用于在Python中提供轻量级流水线的工具。

# joblib具有以下功能：
# 透明的磁盘缓存功能和“懒惰”执行模式，简单的并行计算
# joblib对numpy大型矩阵进行了特定的优化，简单快速

import time, math
from joblib import Parallel, delayed


# 利用joblib实现并行计算
def my_fun(i):
    time.sleep(1)
    return math.sqrt(i ** 2)


num = 10
start = time.time()

for i in range(num):
    my_fun(i)

end = time.time()

print("naive function is :",f'{end - start:.4f}')

# 使用joblib中的Parallel和delayed函数，可以简单的配置my_fun函数的并行运行
# n_jobs : 并行作业的数量

start = time.time()

multi_processing = Parallel(n_jobs=4)

multi_processing(delayed(my_fun)(i) for i in range(num))

end = time.time()

print("After using joblib :",f'{end - start:.4f}')