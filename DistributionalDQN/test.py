import numpy as np

a = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
b = [1, 2, 3, 4]
a = np.vstack(a)
print(a)
print(a.shape)
b = np.array(b)
print(np.sum(np.matmul(a, b), axis=1))
