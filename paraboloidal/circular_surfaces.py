from mayavi import mlab
import numpy as np
from ..functions.paraboloidal_coordinates import pb2cart

# ETS_TOOLKIT=wx

# upper limit for coordinates
up_lim = 6

# constant coordinates
xi_0 = 2
eta_0 = 3
phi_0 = np.pi/4

# point for intersections
x_0, y_0, z_0 = pb2cart(xi_0, eta_0, phi_0)

# step size for coordinates
dxi = 1e-2

# coordinates for constant xi
eta_xi, phi_xi = np.mgrid[0:up_lim+dxi:dxi, 0:2*np.pi+dxi:dxi]
x_xi, y_xi, z_xi = pb2cart(xi_0, eta_xi, phi_xi)

# coordinates for constant eta
xi_eta, phi_eta = np.mgrid[0:up_lim+dxi:dxi, 0:2*np.pi+dxi:dxi]
x_eta, y_eta, z_eta = pb2cart(xi_eta, eta_0, phi_eta)

# coordinates for constant phi
xi_phi, eta_phi = np.mgrid[0:up_lim+dxi:dxi, 0:up_lim+dxi:dxi]
x_phi, y_phi, z_phi = pb2cart(xi_phi, eta_phi, phi_0)

# plot

xi_kwargs = {'colormap': 'spring',
               'opacity': 0.8}
xi1_kwargs = {'colormap': 'hsv',
               'opacity': 0.8}
eta_kwargs = {'colormap': 'winter',
               'opacity': 0.8}
phi_kwargs = {'colormap': 'hsv',
               'opacity': 0.8}

mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0))

## for xi
#mlab.mesh(x_xi, y_xi, z_xi, **xi_kwargs)
#mlab.mesh(-x_xi, y_xi, z_xi, **xi_kwargs)
#mlab.mesh(x_xi, -y_xi, z_xi, **xi_kwargs)
#mlab.mesh(-x_xi, -y_xi, z_xi, **xi_kwargs)

## for eta
#mlab.mesh(x_eta, y_eta, z_eta, **eta_kwargs)
#mlab.mesh(-x_eta, y_eta, z_eta, **eta_kwargs)
#mlab.mesh(x_eta, -y_eta, z_eta, **eta_kwargs)
#mlab.mesh(-x_eta, -y_eta, z_eta, **eta_kwargs)

# for phi
mlab.mesh(x_phi, y_phi, z_phi, **phi_kwargs)

mlab.axes()

# intersection point
#mlab.points3d(x_0, y_0, z_0)

do_save = input('save figure? [y/n]\n')
if do_save == 'y':
    fname = input('input filename\n')
    mlab.savefig(fname + '.png')

mlab.show()

