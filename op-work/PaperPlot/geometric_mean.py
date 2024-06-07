import math

data = [2.01, 1.76, 1.67, 1.69, 1.78, 1.73, 2.03, 1.79, 1.76, 1.70]
def geometric_mean(data):
    total = 1
    for i in data:
        total *= i
    return pow(total,1/len(data))
    
    
print("几何平均值为：",geometric_mean(data))
