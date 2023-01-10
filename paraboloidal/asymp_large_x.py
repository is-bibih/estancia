import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from ..functions.beams import parabV, parabS, parabW

n = 2
m = 2

x = np.linspace(10, 80, num=200).reshape([1,-1])

# expected asymptotic behavior

S_expected = np.exp(1j * np.pi/2 * (m/2 + n)) / gamma(n + 1) \
    * x**(m/2+n) * np.exp(-1j*x/2)
V_expected = 1j * 1/np.pi * gamma(m + n + 1) \
    * np.exp(1j*np.pi/2 * (-m/2 + n)) \
    * np.exp(1j*x/2) \
    * x**(-m/2 - n - 1)

S_expected = S_expected.flatten()
V_expected = V_expected.flatten()

# function

S_asymp = parabS(n, m, 1j*x, ).astype(complex).flatten()
V_asymp = parabV(n, m, 1j*x, ).astype(complex).flatten()
x = x.flatten()

# plot for S

fig, ax = plt.subplots(1,1)
fig.set_size_inches(w=2.5, h=2)
ax.plot(x, x**(-3) * np.real(S_expected), c='black', linestyle='dashed')
ax.plot(x, x**(-3) * np.imag(S_expected), c='red', linestyle='dashed')
ax.plot(x, x**(-3) * np.real(S_asymp), c='black', lw=0.5)
ax.plot(x, x**(-3) * np.imag(S_asymp), c='red', lw=0.5)
ax.set_xlabel('$x$')
ax.set_ylabel(r'$S_\nu^\mu$')
plt.savefig('S_large_x.pdf', bbox_inches='tight')
plt.show()


# plot for V

fig, ax = plt.subplots(1,1)
fig.set_size_inches(w=2.5, h=2)
ax.plot(x, np.real(V_expected), c='black', linestyle='dashed')
ax.plot(x, np.imag(V_expected), c='red', linestyle='dashed')
ax.plot(x, np.real(V_asymp), c='black', lw=0.5)
ax.plot(x, np.imag(V_asymp), c='red', lw=0.5)
ax.set_xlabel('$x$')
ax.set_ylabel(r'$V_\nu^\mu$')
#plt.savefig('V_large_x.pdf', bbox_inches='tight')
plt.show()

