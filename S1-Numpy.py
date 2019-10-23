import numpy as np

# 创建一个Numpy数组
x = np.array([1, 2, 3])
print(x)
print(type(x))
print("========")

# Numpy数组的加减乘除
x = np.array([1, 2, 3])
y = np.array([2, 3, 6])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x / 2)
print("========")

# 创建多维数组
A = np.array([[1, 2], [3, 4]])
print(A)
# 打印多维数组的形状
print(A.shape)
print(A.dtype)
print("========")

# 矩阵的元素相乘
B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)
print(A * 10)
print("========")

# Numpy的广播，即把数组进行自动扩充
A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 20]])
print(A * B)
print("========")

# 访问矩阵元素
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X[0])
print(X[0][1])
# 把矩阵转化为一位数据
X = X.flatten()
print(X)
# 生成数据检索数组
print(X[np.array([0, 2, 4])])
# 获取指定条件的数据
print(X[X > 15])
