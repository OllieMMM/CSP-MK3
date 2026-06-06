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

def nextDir(prevDir):
        """
        Returns the correct search order given a previous direction for the border tarce algorithm
        |7   0   1|
        |6   x   2|
        |5   4   3|
        """
        start_index = (prevDir + 6) % 8
        return [
            start_index % 8,
            (start_index + 1) % 8,
            (start_index + 2) % 8,
            (start_index + 3) % 8,
            (start_index + 4) % 8,
            (start_index + 5) % 8,
            (start_index + 6) % 8,
            (start_index + 7) % 8
        ]

for i in range(8):
    print(f"For index {i} next direction set = {nextDir(i)}")

DIR = np.ndarray([
    [6, 7, 0, 1, 2, 3, 4, 5],
    [7, 0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7, 0],
    [2, 3, 4, 5, 6, 7, 0, 1],
    [3, 4, 5, 6, 7, 0, 1, 2],
    [4, 5, 6, 7, 0, 1, 2, 3],
    [5, 6, 7, 0, 1, 2, 3, 4]
])
