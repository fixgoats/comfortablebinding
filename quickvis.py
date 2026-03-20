import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy as sp
from matplotlib.colors import LogNorm
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("files", nargs="+", default=[])
args = parser.parse_args()

def sqNorm(x):
    return x.real * x.real + x.imag * x.imag

for fname in args.files:
    f = h5py.File(fname, "r")
    psis = f["psi"][:,:]
    points = f["points"][:,:]
    
    fig, ax = plt.subplots()
    fig.suptitle(f"time scan: {os.path.basename(fname)[:-3]}")
    psimax = np.max(sqNorm(psis))
    s = 80 * sqNorm(psis[:,-1]) / psimax
    im = ax.scatter(points[:,0], points[:,1], c=np.angle(psis[:,-1]), s=s, cmap="twilight")
    ax.set_title(f"Params: ")
    plt.colorbar(im)
    fig.savefig(fname[:-3] + ".png")
    plt.cla()
