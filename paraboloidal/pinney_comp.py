import numpy as np
import matplotlib.pyplot as plt

from ..functions.special import P

m = 5
n = 4

r_start = 1
r_end = 10

r = np.linspace(r_start, r_end, num=200).reshape([1,-1])
P_know = laguerreP(m, n, r, method='known_functions').astype(complex).flatten()
P_sums = laguerreP(m, n, r, method='sums').astype(complex).flatten()
#P_inte = laguerreP(m, n, r, method='integrals').astype(complex).flatten()
#P_recu = laguerreP(m, n, r, method='recursive').astype(complex).flatten()

r = r.flatten()

fig, ax = plt.subplots(1,1)
ax.plot(r, np.abs(P_know), label='known functions')
ax.plot(r, np.abs(P_sums), label='sums')
#ax.plot(r, np.abs(P_inte), label='integrals')
#ax.plot(r, np.abs(P_recu), label='recursive')
ax.set_xlabel('$x$')
ax.set_ylabel('$U_n^m(x)$')
fig.legend()
plt.show()

