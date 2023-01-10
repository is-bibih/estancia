import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

from ..functions.special import P

m = 3
n = 2

x = np.linspace(0, 10, num=200).reshape([1,-1])
dx = x[0,1] - x[0,0]

F1 = lambda x: laguerreP(m, n, 1j*x).astype(complex)
F2 = lambda x: laguerreP(m, n, -1j*x).astype(complex)

# evaluate wronskian numerically
wron = (F1(x) * derivative(F2, x, dx=dx) \
        - derivative(F1, x, dx=dx) * F2(x)).flatten()

x = x.flatten()

fig, ax = plt.subplots(1,1)
ax.plot(x, np.abs(wron))
ax.set_ylim(-10, 10)
plt.show()

