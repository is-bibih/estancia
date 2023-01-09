import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.misc import derivative

from ..functions.hypergeometric import pinney_wave

#matplotlib.use("pgf")
#matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#})

n = 2
m = 0

x = np.linspace(0, 5)
ys = pinney_wave(n, m, 1j*x, kind='S')
yv = pinney_wave(n, m, 1j*x, kind='V')

fig, ax = plt.subplots(1,1)
fig.set_size_inches(w=3.5, h=3)
ax.plot(x, np.real(ys), c='black', linestyle='dashed')
ax.plot(x, np.imag(ys), c='black')
ax.plot(x, np.real(yv), c='red')
ax.plot(x, np.imag(yv), c='red', linestyle='dashed')
ax.set_xlabel('$x$')
ax.set_ylim([-10, 30])
plt.savefig('aaaaaa.pdf', bbox_inches='tight')
plt.show()
