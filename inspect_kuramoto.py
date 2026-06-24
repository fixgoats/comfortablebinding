import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File("kuramoto.h5")
a = f["thetas"][:,:]
t = f["times"][:]
oms = f["omegas"][:]

print(np.mean(oms))
order_param = np.mean(np.exp(1j * a), axis = 0)
fig, ax = plt.subplots(2)
r = np.absolute(order_param)
rel_diff = 1 - (r[-2] / r[-1])
print(rel_diff)
ax[0].plot(t, r)
ax[1].plot(t, np.angle(order_param))
plt.show()
