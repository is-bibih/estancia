import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from beams import plane, gaussian
from propagation import beam_propagation_method

# -----------------
# window parameters
# -----------------

x0 = -2e-2
xf = -x0
y0, yf = x0, xf
num = 2**8

# -------------------
# physical parameters
# -------------------

lamb = 633e-9

# for plane wave
k = np.array([0, np.sqrt(0.4), np.sqrt(0.6)]) * 2*np.pi / lamb
r_aperture = (xf - x0)/20

# for gaussian beam
w0 = r_aperture # waist size for gaussian beam

# propagation distance (in terms of z0)
zf = 2 * np.pi*w0**2/lamb

# ------------------------
# initialize and propagate
# ------------------------

x = np.linspace(x0, xf, num=num).reshape([1, num])
y = np.linspace(y0, yf, num=num).reshape([num, 1])

# initial fields

E0p = plane(k, x, y, 0)
E0p[x**2 + y**2 > r_aperture**2] = 0 # put through hole

E0g = gaussian(x, y, 0, r_aperture, lamb)

# fields at zf plane

Efp = beam_propagation_method(E0p, x, y, zf, lamb=lamb)
Efg = beam_propagation_method(E0g, x, y, zf, lamb=lamb)

# plot

imshow_kwargs = {
    'origin': 'lower',
    'cmap': 'inferno',
    'extent': [x0, xf, y0, yf],
}

E = E0g
Ef = Efg

ax = plt.subplot(2, 1, 1)
im = plt.imshow(np.abs(E)**2, **imshow_kwargs)
#im = plt.imshow(np.angle(A0), **imshow_kwargs)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

ax = plt.subplot(2, 1, 2)
im = plt.imshow(np.abs(Ef)**2, **imshow_kwargs)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

print(max(np.abs(Ef.flatten())**2))

plt.show()

