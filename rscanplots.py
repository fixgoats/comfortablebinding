import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.colors import LogNorm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

def sqNorm(x):
    return x.real * x.real + x.imag * x.imag


f = h5py.File(args.file, "r")
psis = f["psis"][0,0,0,:,:].T
rs = f["rscales"][:]
points = f["points"][:,:]
couplings = f["couplings"][:,:]

ci, cj = couplings[0,:]
d = np.linalg.norm(points[ci,:] - points[cj, :])

print(np.shape(psis))
fig, ax = plt.subplots(2, 2)
im1 = ax[0,0].imshow(sqNorm(psis), aspect="auto", interpolation="none", origin="lower", extent=(rs[0], rs[-1], 0, np.shape(psis)[1]), norm=LogNorm())
cb1 = plt.colorbar(im1)
im2 = ax[0, 1].imshow(np.angle(psis), aspect="auto", interpolation="none", cmap="twilight", origin="lower", extent=(rs[0], rs[-1], 0, np.shape(psis)[1]))
cb2 = plt.colorbar(im2)
ax[1, 0].plot(rs*d, sp.special.jv(0, rs*d))
ax[1, 1].plot(rs*d, sp.special.yv(0, rs*d))
ax[1,0].grid()
ax[1,1].grid()

plt.show()
