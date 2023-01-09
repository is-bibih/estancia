import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

from ..functions.paraboloidal import cart2pb
from ..functions.hypergeometric import pinney_wave
from ..functions.propagation import beam_propagation_method

num = 100

n = 3
m = 2

# -----------------
# window parameters
# -----------------

x0 = -0.005
xf = -x0
y0, yf = x0, xf
num = 2**8 + 1
nz = 2**6

z0 = 0

# -------------------
# physical parameters
# -------------------

lamb = 633e-9
r_aperture = (xf - x0)*0.25
beta = 2*np.pi / lamb
alpha = 0.9 * 2*np.pi / r_aperture

print(f'beta = {beta:2f}')
print(f'alpha = {alpha:2f}')
zf = 0.5
print(f'zf = {zf:.2f}')

# ------------------------
# initialize and propagate
# ------------------------

x = np.linspace(x0, xf, num=num).reshape([1, num, 1])
y = np.linspace(y0, yf, num=num).reshape([num, 1, 1])
z = np.linspace(0, zf, num=nz).reshape([1, 1, nz])

xi, eta, phi = cart2pb(x, y, z0)

# initial fields

e01 = pinney_wave(n, m, -1j*xi, kind='V')
e02 = pinney_wave(n, m, 1j*eta, kind='V')
e03 = np.exp(1j*m*phi)

E0 = e01 * e02 * e03
E0[round(num/2), round(num/2), :] = 1e5 # remove singularity
#E0[x**2 + y**2 > r_aperture**2] = 0 # put through hole

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

max_value = np.amax(np.abs(Ef))**2 * 30

fig = plt.figure()
ax = plt.subplot(1,1,1)
im = plt.imshow(np.abs(E0)**2, vmax=max_value, **imshow_kwargs)
plt.xlim(-r_aperture*2, r_aperture*2)
plt.ylim(-r_aperture*2, r_aperture*2)

anim_func = lambda frame: im.set_data(np.abs(E[:,:,frame])**2)

anim = FuncAnimation(fig, anim_func, frames=nz, interval=150)
#anim.save('VmxiVpeta.gif')
plt.show()

