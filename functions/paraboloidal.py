import numpy as np

def gen_pb2cart(mu, nu, lamb, b, c):
    """Transform general paraboloidal coordinates to cartesian.

    :mu: paraboloidal coordinate mu
    :nu: paraboloidal coordinate nu
    :lamb: paraboloidal coordinate lambda
    :b: paraboloidal parameter b
    :c: paraboloidal parameter c
    :returns: (x, y, z) coordinates corresponding to (mu, nu, lambda)

    """
    #if not mu > b > lamb > c > nu > 0:
    #    raise ValueError('arguments should satisfy mu > b > lamb > c > nu > 0')

    x = np.sqrt( 4/(b-c) * (mu-b) * (b-nu) * (b-lamb) )
    y = np.sqrt( 4/(b-c) * (mu-c) * (c-nu) * (lamb-c) )
    z = mu + nu + lamb - b - c
    return x, y, z

def pb2cart(xi, eta, phi):
    """Transform circular paraboloidal coordinates to cartesian.

    :xi: paraboloidal coordinate xi
    :eta: paraboloidal coordinate eta
    :phi: angular coordinate phi
    :returns: (x, y, z) coordinates corresponding to (xi, eta, phi)

    """
    x = xi * eta * np.cos(phi)
    y = xi * eta * np.sin(phi)
    z = -xi + eta
    return x, y, z

def cart2pb(x, y, z):
    """Transform cartesian coordinates to circular paraboloidal.

    :x: cartesian coordinate x
    :y: cartesian coordinate y
    :z: cartesian coordinate z
    :returns: (xi, eta, phi) coordinates corresponding to (x, y, z)

    """
    phi = np.arctan2(y, x)
    xi = - np.sqrt(np.sqrt(x**2 + y**2 + z**2) - z)
    eta = x / (xi * np.cos(phi))
    return xi, eta, phi

