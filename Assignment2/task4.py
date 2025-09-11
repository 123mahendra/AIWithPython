
#####Task 4#####

import numpy as np

A = np.array([[1, 2, 3], [0,1,4], [5,6,0]])

inverseOfA = np.linalg.inv(A)

I1 = np.dot(A, inverseOfA)
I2 = np.dot(inverseOfA, A)

print("Matrix A:")
print(A)

print("\nInverse of A:")
print(inverseOfA)

print("\nA*inverseOfA:")
print(I1)

print("\ninverseOfA*A:")
print(I2)

print("\nCheck:")
print("A*inverseOfA is identity:", np.allclose(I1, np.eye(3)))
print("inverseOfA*A is identity:", np.allclose(I2, np.eye(3)))