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
fig, ax = plt.subplots(3, 2)
fig.set_dpi(300)
fig.set_size_inches(12, 10)
# ax[1, 0].set_title("$J_0(r\\cdot d), Y_0(r\\cdot d)$")
ax[1, 0].set_xlabel("rscale")
ax[0, 0].set_title("$|\\psi_n|^2$, logscale")
ax[0, 1].set_title("$|Arg(\\psi_n*\\bar{\\psi_0})|$")
# ax[1, 0].set_xmargin(0)
ax[1, 1].set_title("Average phase difference")
ax[1, 1].set_xlabel("rscale")
ax[1, 1].set_ylim(bottom=-0.1, top=np.pi+0.1)
# ax[1, 1].set_xmargin(0)
ax[1,0].grid()
for fname in args.files:
    f = h5py.File(fname, "r")
    fig.suptitle(f"rscale time scan: {os.path.basename(fname)[:-3]}")
    snapshot = f["psisnapshot"][0,0,0,:,:,].T
    psiseries = f["sumpsitimeseries"][0,0,0,:,:,0]
    times = f["time"][:]
    rs = f["rscales"][:]
    points = f["points"][:,:]
    couplings = f["couplings"][:,:]
    t0 = times[0]
    t01 = times[1]
    t02 = times[-2048]
    t1 = times[-1]
    period = t1 - t02
    dt = t01 - t0
    emax = np.pi / dt 
    de = 2 * np.pi / period
    print(type(dt))
    energies = np.arange(-emax, emax, de)
    couplephasediffs = np.array([np.mean(np.array([np.abs(np.angle(snapshot[ci, i]*snapshot[cj, i].conj())) for ci, cj in couplings[:]])) for i, _ in enumerate(rs)])
    
    unique_coupling_lengths = []
    for ci, cj in couplings[:]:
        d = np.linalg.norm(points[ci, :] - points[cj, :])
        if not any(abs(x - d) < 0.05 for x in unique_coupling_lengths):
            unique_coupling_lengths.append(d)
    
    print(unique_coupling_lengths)
    im1 = ax[0,0].imshow(sqNorm(snapshot), aspect="auto", interpolation="none", origin="lower", extent=(rs[0], rs[-1], 0, np.shape(snapshot)[1]), norm=LogNorm())
    cb1 = plt.colorbar(im1)
    im2 = ax[0, 1].imshow(np.abs(np.angle(snapshot)), aspect="auto", interpolation="none", cmap="inferno", origin="lower", extent=(rs[0], rs[-1], 0, np.shape(snapshot)[1]))
    cb2 = plt.colorbar(im2)
    ax[1, 0].set_xlim(left=rs[0], right=rs[-1])
    ax[1, 1].set_xlim(left=rs[0], right=rs[-1])
    lowest_d = np.min(unique_coupling_lengths)
    jymax = max(np.max(sp.special.jv(0, rs*lowest_d)), np.max(sp.special.yv(0, rs*lowest_d)))
    jymin = min(np.min(sp.special.jv(0, rs*lowest_d)), np.min(sp.special.yv(0, rs*lowest_d)))
    jydelta = jymax - jymin
    ax[1,0].set_ylim(bottom=jymin-0.02*jydelta, top=jymax+0.02*jydelta)
    lines = []
    for d in unique_coupling_lengths:
        j = sp.special.jv(0, rs*d)
        y = sp.special.yv(0, rs*d)
        line1, = ax[1, 0].plot(rs, j, label=f"$J_0(rscale\\cdot {d:.3f})$")
        line2, = ax[1, 0].plot(rs, y, label=f"$Y_0(rscale\\cdot {d:.3f})$")
        lines.append(line1)
        lines.append(line2)
    ax[1,0].legend()
    line3, = ax[1, 1].plot(rs, couplephasediffs, color="green")
    psifft = np.fft.fftshift(np.fft.ifft(psiseries[:,-2048:], norm="ortho"))
    psifftsqnorm = sqNorm(psifft)
    im3 = ax[2,0].imshow(sqNorm(psiseries[:,::10].T), extent=(rs[0], rs[-1], t0, t1), aspect="auto", norm=LogNorm(), origin="lower", interpolation="none")
    im4 = ax[2,1].imshow(psifftsqnorm.T[1024-64:1024+64,:], extent=(rs[0], rs[-1], energies[1024-64], energies[1024+64]), origin="lower", aspect="auto", norm=LogNorm(),interpolation="none")
    cb3 = plt.colorbar(im3)
    cb4 = plt.colorbar(im4)
    
    fig.savefig(fname[:-3] + ".png")
    cb1.remove()
    cb2.remove()
    line3.remove()
    for line in lines:
        line.remove()
    cb3.remove()
    cb4.remove()
