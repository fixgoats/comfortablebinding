import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy as sp
from matplotlib.colors import LogNorm
from matplotlib.animation import FFMpegWriter
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("files", nargs="+", default=[])
args = parser.parse_args()

def sqNorm(x):
    return x.real * x.real + x.imag * x.imag

writer = FFMpegWriter(fps=30)
for fname in args.files:
    f = h5py.File(fname, "r")
    psis = f["psi"][:,:]
    points = f["points"][:,:]
    params = f["params"]
    
    fig, ax = plt.subplots()
    fig.suptitle(f"time scan: {os.path.basename(fname)[:-3]}, rscale: {params["rscale"]}")
    psimax = np.max(sqNorm(psis))
    scatter = ax.scatter(points[:,0], points[:,1], c= np.zeros(np.shape(points)[0]), cmap="twilight")
    scatter.set_clim(vmin=-np.pi, vmax=np.pi)
    cb = fig.colorbar(scatter)
    with writer.saving(fig, f"{fname[:-3]}.mp4", dpi=200):

        for i in range(0, np.shape(psis)[1], 100):
            s = 80 * sqNorm(psis[:,i]) / psimax
            scatter.set_sizes(s)
            scatter.set_array(np.angle(psis[:,i] * psis[0,i].conj()))
            writer.grab_frame()

    plt.cla()
