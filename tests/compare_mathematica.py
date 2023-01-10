import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser

from ..functions.special import X

fname = expanduser("~/grive/ifi7/optica/mathematica/datos/laguerreX.csv")
datos = np.genfromtxt(fname, delimiter=',')

m = 2
n = 35

x = datos[:, 0]
lagX_ma = datos[:, 1]
lagX_py = laguerreX(m, n, x.reshape([1,-1])).flatten()
err = (lagX_ma - lagX_py)/lagX_ma

fig, ax = plt.subplots(1,2)
ax[0].plot(x, lagX_ma, label='mathematica')
ax[0].scatter(x, lagX_py, label='python')
ax[0].set_ylim(-40, 25)
ax[1].plot(x, err, label='error relativo a mathematica')
fig.legend()
plt.show()

