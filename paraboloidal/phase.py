import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ..functions.paraboloidal_coordinates import cart2pb
from ..functions.special import parabV, parabS, parabW

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 9,
})

num = 200

n = 3
ms = np.arange(start=-1, stop=5)

z = 0

x0 = -2
xf = -x0
y0 = x0
yf = -y0

x = np.linspace(x0, xf, num=num).reshape([1, -1])
y = np.linspace(y0, yf, num=num).reshape([-1, 1])

xi, eta, phi = cart2pb(x, y, z)

# transverse distribution

fig, ax = plt.subplots(3,2)
fig.set_size_inches(w=5, h=9)
im = None

ax = ax.flatten()

for i in range(len(ms)):

    title = r"$\mu = " + str(int(ms[i])) + r"$"

    dist = parabS(n, ms[i], -1j*xi, ) \
        * parabS(n, ms[i], 1j*eta, ) \
        * np.exp(1j*ms[i]*phi)

    phase = np.angle(dist)

    im = ax[i].imshow(phase, cmap='inferno', origin='lower',
                      extent=(x0, xf, y0, yf),
                      vmin=-np.pi, vmax=np.pi)

    ax[i].set_title(title)

    print('done ' + str(i+1))

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

plt.savefig('phase.pdf')

