import numpy as np

a = [[i, j] for i in range(100) for j in range(100)]
np.savetxt("square100x100.txt", a)
