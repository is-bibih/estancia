import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ..functions.paraboloidal import cart2pb
from ..functions.hypergeometric import pinney_wave

#matplotlib.use("pgf")
#matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#    'font.size': 9,
#})

num = 100

n = 3
m = 2

z = 0

x0 = -1
xf = -x0
y0 = x0
yf = -y0

#kind1 = ['S', 'V', 'S', 'S']
#kind2 = ['V', 'S', 'S', 'S']
#sign1 = [-1, 1, 1, -1]
#sign2 = [1, -1, -1, 1]
#
#fname = 'transverse_dist_bright.pdf'

#kind1 = ['S', 'V', 'V', 'V']
#kind2 = ['V', 'S', 'V', 'V']
#sign1 = [1, -1, -1, 1]
#sign2 = [-1, 1, 1, -1]
#
#fname = 'transverse_dist_dim.pdf'

kind1 = ['S', 'W', 'S', 'S', 'S', 'W', 'W', 'W']
kind2 = ['W', 'S', 'S', 'S', 'W', 'S', 'W', 'W']
sign1 = [-1, 1, 1, -1, 1, -1, -1, 1]
sign2 = [1, -1, -1, 1, -1, 1, 1, -1]

x = np.linspace(x0, xf, num=num).reshape([1, -1])
y = np.linspace(y0, yf, num=num).reshape([-1, 1])

fname = 'transverse_dist_bright.pdf'

xi, eta, phi = cart2pb(x, y, z)

# transverse distribution

#fig, ax = plt.subplots(2,2)
#fig.set_size_inches(w=5, h=6)
fig, ax = plt.subplots(4,2)
fig.set_size_inches(w=5, h=12)
im = None

ax = ax.flatten()

for i in range(len(kind1)):

    first_sign = "+" if sign1[i] == 1 else "-"
    second_sign = "+" if sign2[i] == 1 else "-"
    title = "$" + kind1[i] + r"_\nu^{\mu} (" + first_sign + r"i\xi) " \
        + kind2[i] + r"_\nu^{\mu} (" + second_sign + r"i\eta) $"

    dist = pinney_wave(n, m, sign1[i]*1j*xi, kind=kind1[i]) \
        * pinney_wave(n, m, sign2[i]*1j*eta, kind=kind2[i]) \
        * np.exp(1j*m*phi)

    I = np.abs(dist)

    im = ax[i].imshow(I, cmap='inferno', origin='lower',
                      extent=(x0, xf, y0, yf),
                      vmin=0)

    ax[i].set_title(title)

    print('done ' + str(i+1))

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

#plt.savefig(fname)
plt.show()

