import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

from beams import bessel, plane, gaussian
from propagation import beam_propagation_method

# -----------------
# window parameters
# -----------------

x0 = -0.02
xf = -x0
y0, yf = x0, xf
num = 2**9 + 1
nz = 2**6

# -------------------
# physical parameters
# -------------------

lamb = 633e-9
r_aperture = (xf - x0)/4
#r_aperture = 0.001
beta = 2*np.pi / lamb
alpha = 0.9 * 2*np.pi / r_aperture
#alpha = 0.001 * beta
print(f'beta = {beta:2f}')
print(f'alpha = {alpha:2f}')
zf = r_aperture * np.sqrt((2*np.pi/(alpha*lamb))**2 - 1)
print(f'zf = {zf:.2f}')

# ------------------------
# initialize and propagate
# ------------------------

x = np.linspace(x0, xf, num=num).reshape([1, num, 1])
y = np.linspace(y0, yf, num=num).reshape([num, 1, 1])
z = np.linspace(0, zf, num=nz).reshape([1, 1, nz])
#x = np.linspace(x0, xf, num=num).reshape([1, num])
#y = np.linspace(y0, yf, num=num).reshape([num, 1])

# initial fields

E0 = bessel(x, y, 0, alpha, beta)
#E0 = plane([0, 0, np.pi*2 / lamb], x, y, 0)
#E0 = gaussian(x, y, 0, 0.002, lamb)
E0[x**2 + y**2 > r_aperture**2] = 0 # put through hole

E = beam_propagation_method(E0, x, y, z, lamb=lamb)
Ef = E[:, :, -1]

# plot

imshow_kwargs = {
    'origin': 'lower',
    'cmap': 'inferno',
    'extent': [x0, xf, y0, yf],
}

ax = plt.subplot(2, 1, 1)
im = plt.imshow(np.abs(E0)**2, **imshow_kwargs)
#im = plt.imshow(np.angle(A0), **imshow_kwargs)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

ax = plt.subplot(2, 1, 2)
im = plt.imshow(np.abs(Ef)**2, **imshow_kwargs)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.show()

# propagation animation

fig = plt.figure()
ax = plt.subplot(1,1,1)
im = plt.imshow(np.abs(E0)**2, **imshow_kwargs)
plt.xlim(-r_aperture*2, r_aperture*2)
plt.ylim(-r_aperture*2, r_aperture*2)

anim_func = lambda frame: im.set_data(np.abs(E[:,:,frame])**2)

anim = FuncAnimation(fig, anim_func, frames=nz, interval=150)
plt.show()

