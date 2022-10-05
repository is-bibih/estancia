import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

from ..functions.beams import bessel, hankel
from ..functions.propagation import beam_propagation_method

# -----------------
# window parameters
# -----------------

x0 = -0.02
xf = -x0
y0, yf = x0, xf
num = 2**9# + 1
num_z = 300

# -------------------
# physical parameters
# -------------------

lamb = 633e-9

r_aperture = (xf - x0)*0.25
beta = 2*np.pi / lamb
alpha = 0.9 * 2*np.pi / r_aperture
zf = r_aperture * np.sqrt((2*np.pi/(alpha*lamb))**2 - 1)

m = 0
kz = 2*np.pi/lamb
kr = 3e-4 * kz
varphi = np.arctan(kr/kz)

obstruction = False
r_obs = (xf - x0)*0.01
print(r_obs)
obs_fun = lambda x, y: x.reshape([1, num])**2 + y.reshape([num, 1])**2 < (r_obs)**2
obs_z = 0.05*zf
print(f'obstruction at {obs_z:2f}')

# ------------------------
# initialize and propagate
# ------------------------

x = np.linspace(x0, xf, num=num).reshape([1, num, 1])
y = np.linspace(y0, yf, num=num).reshape([num, 1, 1])
z = np.linspace(0, zf, num=num_z).reshape([1, 1, num_z])

# make initial field and propagate

E0 = hankel(x, y, 0, kr, kz, m=m, kind=2)

if obstruction:

    # placeholder field
    E = np.zeros([num, num, num_z], dtype=np.cdouble)

    # z before and after obstruction
    idx_before = z < obs_z
    idx_after = z >= obs_z
    z_before = z[idx_before]
    z_after = z[idx_after]
    z_after = z_after - z_after.flatten()[0]

    # propagate to obstruction
    E[:,:,:z_before.size] = beam_propagation_method(E0, x, y, z_before, lamb=lamb)

    # obstruct
    E_obs = E[:,:,z_before.size-1]
    E_obs[obs_fun(x, y)] = 0
    E[:,:,z_before.size] = E_obs
    E_obs = E_obs.reshape([num, num, 1])

    # propagate after obstruction
    E[:,:,z_before.size:] = beam_propagation_method(E_obs, x, y, z_after, lamb=lamb)

else:
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

# intensity at center

idx_center = int(np.floor(num/2)+1)
E_center = E[idx_center, idx_center, :]

ax = plt.subplot(1, 1, 1)
ax.plot(z.flatten(), np.abs(E_center)**2)
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$I(r=0, z)$')
plt.show()

# propagation animation

fig = plt.figure()
ax = plt.subplot(1,1,1)
im = plt.imshow(np.abs(E0)**2, **imshow_kwargs)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title(r'$z=0$')
plt.xlim(-r_aperture*2, r_aperture*2)
plt.ylim(-r_aperture*2, r_aperture*2)

def anim_func(frame):
    z_current = z.flatten()[frame]
    im.set_data(np.abs(E[:,:,frame])**2)
    ax.set_title(f'$z={z_current:.2f}$')

anim = FuncAnimation(fig, anim_func, frames=num_z, interval=100)
plt.show()

