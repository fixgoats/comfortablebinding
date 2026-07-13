import numpy as np

grid = np.array([[i, j] for i in range(15) for j in range(15)])
np.savetxt("square225.txt", grid)
