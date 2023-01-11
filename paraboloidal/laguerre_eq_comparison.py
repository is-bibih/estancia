import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.misc import derivative
from ..functions.paraboloidal_coordinates import cart2pb, pb2cart
from ..functions.special import laguerreP, laguerreX, laguerreL, parabV, parabS, parabW

m = 0
n = 5

# compare pinney's function to X in laguerre equation

r_start = 1
r_end = 20
r = np.linspace(r_start, r_end, num=500).reshape([1,-1])
dr = r[0,1] - r[0,0]
dr = float(dr)

X_eval = laguerreX(m, n, r, no_factorials=True).flatten()
P_eval = laguerreP(m, n, r, method='known_functions').astype(complex).flatten()
L_eval = laguerreL(n, m, r).flatten().flatten()

r = r.flatten()

# laguerre equation
lag_eq = lambda u: r * np.gradient(u, dr, edge_order=2) \
        + (m + 1 - r) * np.gradient(u, dr, edge_order=1) \
        + n * u

lag_X = lag_eq(X_eval)
lag_P = lag_eq(P_eval)
lag_L = lag_eq(L_eval)

fig, ax = plt.subplots(1,1)
ax.plot(r, lag_X, label='X')
ax.plot(r, np.abs(lag_P), label='P')
ax.plot(r, lag_L, label='L')
fig.legend()
plt.show()

