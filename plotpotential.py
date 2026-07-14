import numpy as np
import matplotlib.pyplot as plt

pts = np.loadtxt("sample_monopts.txt")
#pts = np.array([[3, 3], [4, 4], [7, 3]])
xmin = np.min(pts[:,0])
ymin = np.min(pts[:,1])
xmax = np.max(pts[:,0])
ymax = np.max(pts[:,1])
mask = (pts[:,0] >= 0) & (pts[:,1] >= 0) & (pts[:, 0] <= 5) & (pts[:, 1] <= 5)
pts = pts[mask]

V = np.zeros((1024, 1024))
x = np.linspace(0, 5, 1024)
xx, yy = np.meshgrid(x, x)

for p in pts:
    V += np.exp(-(((xx - p[0])**2 + (yy - p[1])**2)) / 0.01)

fig, ax = plt.subplots()
# ax.scatter(pts[:,0], pts[:,1], s=8, c="orange")
ax.imshow(V, extent=[0, 10, 0, 10], origin="lower")
fig.savefig("potential.pdf")
