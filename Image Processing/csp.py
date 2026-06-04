import numpy as np

A = np.array([
    [[1], [2], [3]],
    [[2], [2], [6]],
    [[2], [2], [8]],
    [[2], [2], [11]]
    ])

print(A.shape)

h, w, _ = A.shape

paddedA = np.insert(A, [0, h], 0, axis=0)
paddedA = np.insert(paddedA, [0, w], 0, axis=1)

maskA = np.array(paddedA == 2)

print(paddedA)

print(maskA)