import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from ..functions.paraboloidal import cart2pb, pb2cart
from ..functions.hypergeometric import P, X, L, pinney_wave

m = 4
n = 3

# compare pinney's function to X
r_start = 1
r_end = 10
r = np.linspace(r_start, r_end).reshape([1,-1])

coef_X = (-1)**(n+1) / (factorial(n) * factorial(m))
X_eval = X(m, n, r)
P_eval = P(m, n, r).astype(complex)

r, P_eval, X_eval = [xi.flatten() for xi in (r, P_eval, X_eval)]

fig, ax = plt.subplots(1,1)
ax.plot(r, X_eval * coef_X, label='Laguerre de 2o tipo')
ax.scatter(r, np.imag(P_eval), label=r'Im{Pinney}')
plt.legend()
plt.show()

