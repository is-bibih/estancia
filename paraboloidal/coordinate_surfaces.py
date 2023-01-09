from mayavi import mlab
import numpy as np
from ..functions.paraboloidal import gen_pb2cart

# ETS_TOOLKIT=wx

# parameters
b = 20
c = 10

# upper limit for coordinates
up_lim = (b+c)*1.2

# constant coordinates
mu_0 = up_lim * 0.9/1.2
mu_1 = mu_0 * 1.1
nu_0 = c/2 * 0.9
lamb_0 = (b+c)/2 * 0.9

# point for intersections
x_0, y_0, z_0 = gen_pb2cart(mu_0, nu_0, lamb_0, b, c)

# step size for coordinates
dxi = 1e-2

# coordinates for constant mu
nu_mu, lamb_mu = np.mgrid[0:c+dxi:dxi, c:b+dxi:dxi]
x_mu, y_mu, z_mu = gen_pb2cart(mu_0, nu_mu, lamb_mu, b, c)
x_mu1, y_mu1, z_mu1 = gen_pb2cart(mu_1, nu_mu, lamb_mu, b, c)

# coordinates for constant nu
mu_nu, lamb_nu = np.mgrid[b:up_lim+dxi:dxi, c:b+dxi:dxi]
x_nu, y_nu, z_nu = gen_pb2cart(mu_nu, nu_0, lamb_nu, b, c)

# coordinates for constant lambda
mu_lamb, nu_lamb = np.mgrid[b:up_lim+dxi:dxi, 0:c+dxi:dxi]
x_lamb, y_lamb, z_lamb = gen_pb2cart(mu_lamb, nu_lamb, lamb_0, b, c)

# plot

mu_kwargs = {'colormap': 'spring',
               'opacity': 0.8}
mu1_kwargs = {'colormap': 'hsv',
               'opacity': 0.8}
nu_kwargs = {'colormap': 'winter',
               'opacity': 0.8}
lamb_kwargs = {'colormap': 'summer',
               'opacity': 0.8}

mlab.figure(bgcolor=(1,1,1))

# for mu
mlab.mesh(x_mu, y_mu, z_mu, **mu_kwargs)
mlab.mesh(-x_mu, y_mu, z_mu, **mu_kwargs)
mlab.mesh(x_mu, -y_mu, z_mu, **mu_kwargs)
mlab.mesh(-x_mu, -y_mu, z_mu, **mu_kwargs)

#mlab.mesh(x_mu1, y_mu1, z_mu1, **mu1_kwargs)
#mlab.mesh(-x_mu1, y_mu1, z_mu1, **mu1_kwargs)
#mlab.mesh(x_mu1, -y_mu1, z_mu1, **mu1_kwargs)
#mlab.mesh(-x_mu1, -y_mu1, z_mu1, **mu1_kwargs)

# for nu
mlab.mesh(x_nu, y_nu, z_nu, **nu_kwargs)
mlab.mesh(-x_nu, y_nu, z_nu, **nu_kwargs)
mlab.mesh(x_nu, -y_nu, z_nu, **nu_kwargs)
mlab.mesh(-x_nu, -y_nu, z_nu, **nu_kwargs)

# for lambda
mlab.mesh(x_lamb, y_lamb, z_lamb, **lamb_kwargs)

mlab.axes()

# intersection point
mlab.points3d(x_0, y_0, z_0)

#do_save = input('save figure? [y/n]\n')
#if do_save == 'y':
#    fname = input('input filename\n')
#    mlab.savefig(fname + '.png')

mlab.show()

