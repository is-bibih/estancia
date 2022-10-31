import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

from ..functions.hypergeometric import pinney_wave

n = 4
m = 5
k = 2*np.pi / 633e-9
#k = 1

def diff_eq(kind, z):
    """evaluate given function in differential equation for coordinates

    :kind: kind of pinney wave function to evaluate
    :returns: difference from zero of function evaluated in differential equation

    """
    func = lambda x: pinney_wave(n, m, x, kind=kind)

    diff_eq_func = lambda x: x * derivative(func, z, n=2) \
        + (m + 1 - 2*1j*k*x) * derivative(func, z, n=1) \
        + 2*1j*k*n * func(x)

    return diff_eq_func(z)

# evaluate differential equation
x = np.linspace(1, 10)
ys = diff_eq('S', x).astype(complex).flatten()
yv = diff_eq('V', x).astype(complex).flatten()

# plot
fig, ax = plt.subplots(1, 1)
ax.plot(x, np.abs(ys), label='S')
ax.plot(x, np.abs(yv), label='V')
fig.legend()
plt.show()

