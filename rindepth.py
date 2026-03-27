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

timepixels = 600

for fname in args.files:
    fig, ax = plt.subplots(2,2)

    ax[0, 0].set_title("Average phase difference")
    ax[0, 0].set_xlabel("t")
    ax[0, 1].set_title("$|\\psi_n|^2$")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("n")
    ax[1, 0].set_title("$|Arg(\\psi_n\\bar{\\psi_0}|$")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("n")
    ax[1, 1].set_title("$|ifft(\\psi_n)|^2$")
    ax[1, 1].set_xlabel("$\\omega$")
    ax[1, 1].set_ylabel("$n$")
    fig.suptitle(f"rscale time scan: {os.path.basename(fname)[:-3]}")

    f = h5py.File(fname, "r")
    psiseries = f["psi"][:,:]
    times = f["time"][:]
    couplings = f["couplings"][:,:]
    timestride = np.shape(psiseries)[1] // timepixels

    t0 = times[0][0]
    t01 = times[1][0]
    t02 = times[-2048][0]
    t1 = times[-1][0]
    period = t1 - t02
    dt = t01 - t0
    emax = np.pi / dt 
    de = 2 * np.pi / period
    energies = np.arange(-emax, emax, de)
    couplephasediffs = np.array([np.mean(np.array([np.abs(np.angle(psiseries[ci, ::timestride]*psiseries[cj, ::timestride].conj())) for ci, cj in couplings[:]]), axis=0)])

    ax[0, 0].plot(times[::timestride], couplephasediffs.T)
    
    im1 = ax[0, 1].imshow(sqNorm(psiseries), interpolation="none", origin="lower", aspect="auto", extent=(t0, t1, 0, np.shape(psiseries)[0]))
    im2 = ax[1, 0].imshow(np.abs(np.angle(psiseries[:,::timestride]*psiseries[0,::timestride])), interpolation="none", origin="lower", aspect="auto", extent=(t0, t1, 0, np.shape(psiseries)[0]), cmap="twilight")

    psifft = np.fft.fftshift(np.fft.ifft(psiseries[:,-2048:], norm="ortho"))
    psifftsqnorm = sqNorm(psifft)
    im3 = ax[1,1].imshow(psifftsqnorm.T[1024-64:1024+64,:], extent=(0, np.shape(psiseries)[0], energies[1024-64], energies[1024+64]), origin="lower", aspect="auto", interpolation="none")

    plt.show()
