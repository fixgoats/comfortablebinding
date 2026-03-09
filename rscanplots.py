import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.colors import LogNorm
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("files", nargs="+", default=[])
args = parser.parse_args()

def sqNorm(x):
    return x.real * x.real + x.imag * x.imag

fig, ax = plt.subplots(2, 2)
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
    fig.suptitle(f"rscale scan: {os.path.basename(fname)[:-3]}")
    psis = f["psis"][0,0,0,:,:].T
    rs = f["rscales"][:]
    points = f["points"][:,:]
    couplings = f["couplings"][:,:]

    couplephasediffs = np.array([np.mean(np.array([np.abs(np.angle(psis[ci, i]*psis[cj, i].conj())) for ci, cj in couplings[:]])) for i, _ in enumerate(rs)])
    
    unique_coupling_lengths = []
    for ci, cj in couplings[:]:
        d = np.linalg.norm(points[ci, :] - points[cj, :])
        if not any(abs(x - d) < 0.05 for x in unique_coupling_lengths):
            unique_coupling_lengths.append(d)
    
    print(unique_coupling_lengths)
    im1 = ax[0,0].imshow(sqNorm(psis), aspect="auto", interpolation="none", origin="lower", extent=(rs[0], rs[-1], 0, np.shape(psis)[1]), norm=LogNorm())
    cb1 = plt.colorbar(im1)
    im2 = ax[0, 1].imshow(np.abs(np.angle(psis)), aspect="auto", interpolation="none", cmap="inferno", origin="lower", extent=(rs[0], rs[-1], 0, np.shape(psis)[1]))
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
    
    fig.savefig(fname[:-3] + ".png")
    cb1.remove()
    cb2.remove()
    line3.remove()
    for line in lines:
        line.remove()
