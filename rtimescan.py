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


# fig, ax = plt.subplots(2)
fig, ax = plt.subplots()
f = h5py.File(args.files[0], "r")
fig.suptitle("Bleh")
psis = f["psis"][0,0,0, :,-2048:,0]
times = f["time"][-2048:]
rs = f["rscales"][:]
t0 = times[0]
t01 = times[1]
t1 = times[-1]
period = t1 - t0
dt = t01 - t0
emax = np.pi / dt 
de = 2 * np.pi / period
energies = np.arange(-emax, emax, de)

psifft = np.fft.fftshift(np.fft.fft(psis, norm="ortho"))
psifftsqnorm = sqNorm(psifft)
# ax[0].plot(times, psis.real)
# ax[1].plot(times, psis.imag)
# ax[0].plot(times, psifft.real)
# ax[1].plot(times, psifft.imag)
ax.set_yscale("log")
ax.plot(energies[1024-64:1024+64], psifftsqnorm.T[1024-64:1024+64,:], label=[f"rscale={r}" for r in rs])
ax.legend()
#hist = np.histogram(np.ones(2048), bins=energies, weights=psifftsqnorm)

plt.show()
