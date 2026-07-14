import numpy as np
import matplotlib.pyplot as plt

sqrt3 = np.sqrt(3)
A = np.array([[0, 0],[0, sqrt3], [-1, sqrt3], [-1.5, sqrt3 / 2], [-2, 0], [-1.5, -sqrt3 / 2]])

a1 = np.array([3, -sqrt3])
a2 = np.array([3, sqrt3])
a3 = np.array([0, -2*sqrt3])
# grid = np.concatenate([A + i * a1 + j * a2 for i in range(-4,4) for j in range(-4,4)])
grid = np.concatenate([A + i * a1 +j * a3 for i in range(-40,40) for j in range(-40,40)])

mask = np.linalg.norm(grid, axis=1) < 11.51*2
# print(grid[mask])
# fig, ax = plt.subplots()
# plt.scatter(grid[mask,0], grid[mask,1])
# plt.axis("equal")
# plt.show()
ret = grid[mask]
np.savetxt("largestcirculartetrille.txt", ret)
print(np.shape(ret))
