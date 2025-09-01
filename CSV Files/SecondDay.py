import numpy as np
# from numpy.ma.core import shape
#
# arr = np.array([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]]])
#
# print(arr.ndim)
#
# print(arr)
#
# print(arr[0][2][1][1])
#
# # print(arr[0,2,1,1])
#
# print(shape(arr))

# z = np.ones(10)
# print(z)

# z1 = np.linespace(0,5,0.2)
# print(z1)

# z1 = np.arange(0,5,0.2)
# print(z1)

# z2 = np.random.randint(1,11,100)
# print(z2)

# z3 = np.random.randn(10)
# print(z3)

z4 = np.repeat([[1,2,3]],4,0)
print(z4)

A = np.delete(z4,[2],1)
B = np.delete(z4,[1],0)

print(A)
print(B)




