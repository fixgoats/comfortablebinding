import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as sp
from argparse import ArgumentParser
import os

# parser = ArgumentParser()
# parser.add_argument("files", nargs="+", default=[])
# args = parser.parse_args()

def sqNorm(x):
    return x.real * x.real + x.imag * x.imag


# fig, ax = plt.subplots(2)
# plt.rcParams.update({'font.size': 18})
# plt.tight_layout()
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
# fig.set_c
fig.set_dpi(300)
# fig.set_size_inches(12, 7)
ticklabelsize = 18
axislabelsize = 26
legendfontsize = 22
ax.tick_params(axis="x", labelsize=ticklabelsize)
ax.tick_params(axis="y", labelsize=ticklabelsize)
# ax[1, 0].set_title("$J_0(r\\cdot d), Y_0(r\\cdot d)$")
ax.set_xlabel("shortest coupling length (arb. units)", fontsize=axislabelsize, fontweight="bold")
ax.set_ylabel("${\\bf \\Delta \\Phi}$", fontsize=axislabelsize, fontweight="bold")
files = [("gögn/mono330scaletimescan.h5", "Hat", "lightcoral"), ("gögn/square400scaletimescan.h5", "Square", "firebrick"), ("gögn/tetrille462scaletimescan.h5", "Tetrille", "maroon")]
for fname, label, color in files:
    f = h5py.File(fname, "r")
    # fig.suptitle(f"rscale time scan: {os.path.basename(fname)[:-3]}")
    snapshot = f["psisnapshot"][0,0,0,:,:,].T
    # psiseries = f["sumpsitimeseries"][0,0,0,:,:,0]
    # times = f["time"][:]
    rs = f["rscales"][:]
    points = f["points"][:,:]
    couplings = f["couplings"][:,:]
    # t0 = times[0]
    # t01 = times[1]
    # t02 = times[-2048]
    # t1 = times[-1]
    # period = t1 - t02
    # dt = t01 - t0
    # emax = np.pi / dt 
    # de = 2 * np.pi / period
    # print(type(dt))
    # energies = np.arange(-emax, emax, de)
    couplephasediffs = np.array([np.mean(np.array([np.abs(np.angle(snapshot[ci, i]*snapshot[cj, i].conj())) for ci, cj in couplings[:]])) for i, _ in enumerate(rs)])
    
    unique_coupling_lengths = []
    for ci, cj in couplings[:]:
        d = np.linalg.norm(points[ci, :] - points[cj, :])
        if not any(abs(x - d) < 0.05 for x in unique_coupling_lengths):
            unique_coupling_lengths.append(d)
    print(unique_coupling_lengths)
    lowest_d = np.min(unique_coupling_lengths)
    
    line, = ax.plot(rs * lowest_d, couplephasediffs, label=label, color=color, linewidth=4)
    ax.set_xlim(0.5, 12)
    ax.legend(loc="best", prop={'weight': 'bold', 'size': legendfontsize})

x = np.linspace(0.5, 12, 200)
# jymax = max(np.max(sp.special.jv(0, x)), np.max(sp.special.yv(0, x)))
# jymin = min(np.min(sp.special.jv(0, x)), np.min(sp.special.yv(0, x)))
#jydelta = jymax - jymin
#ax.set_ylim(bottom=jymin-0.02*jydelta, top=jymax+0.02*jydelta)
axh = ax.twinx()
# ax.yaxis.tick_right()
j = sp.special.jv(0, x)
y = sp.special.yv(0, x)
line1, = axh.plot(x, j, label="${{\\bf J_0}}$", color="teal", linewidth=4)
line2, = axh.plot(x, y, label="${{\\bf Y_0}}$", color="darkslategrey", linewidth=4)
axh.tick_params(axis="y", labelcolor="deepskyblue", labelsize=ticklabelsize)
axh.set_ylabel("${{\\bf J_0(x), Y_0(x)}}$", fontsize=axislabelsize, fontweight="bold")
axh.grid()
axh.legend(prop={'weight': 'bold', 'size': legendfontsize})
fig.savefig("phasecomparison.png")
