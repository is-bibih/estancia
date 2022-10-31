import numpy as np
import matplotlib.pyplot as plt

from ..functions.hypergeometric import P

m = 5
n = 50

r_start = 1
r_end = 10

r = np.linspace(r_start, r_end, num=200).reshape([1,-1])
P_eval = P(m, n, r, method='sums').astype(complex)

r = r.flatten()
P_eval = P_eval.flatten()

# asymptotic behavior

ch_sign = np.ones(r.shape)
ch_sign[np.sign(r) != -1] = 1

asymp_coef = (1/np.sqrt(np.pi) \
              * n**(0.5*m - 0.25) * r**(-0.5*m - 0.25) \
              * np.exp(0.5*r))
ang = 2*np.sqrt(n*r) - 0.5*np.pi*m - (3/4)*np.pi
asymp_re = asymp_coef * np.cos(ang)
asymp_im = asymp_coef * np.sin(ang)

comparison_P = P_eval

fig, ax = plt.subplots(1,1)
ax.scatter(r, np.real(comparison_P), label='pinney real', s=1)
ax.plot(r, asymp_re, label='asintotico real')
ax.scatter(r, np.imag(comparison_P), label='pinney imag', s=1)
ax.plot(r, asymp_im, label='asintotico imag')
fig.legend()
plt.show()

