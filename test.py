import numpy as np
sizes = [2, 3, 1]
print(sizes[1:])
print(sizes[:-1])
print(sizes[1:])
print(zip(sizes[:-1],sizes[1:]))
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print(weights)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print(biases)
print(biases[0].shape)