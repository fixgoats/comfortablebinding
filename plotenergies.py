import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
es = np.loadtxt("eigvals.txt")
ax.set_xlabel("N")
ax.set_ylabel("E (arb. units)")
ax.set_title("$t=-\\exp(-|\\mathbf{r}|)$, cutoff at $|\\mathbf{r}|=3.9$")
ax.plot(es)
plt.show()
