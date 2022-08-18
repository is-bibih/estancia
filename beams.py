import numpy as np
from scipy.special import j0

def plane(k, x, y, z):
    """Generate plane wave with wave vector k.

    :k: tuple (kx, ky, kz)
    :x: array of x values
    :y: array of y values
    :z: plane to evaluate wave on
    :returns: complex amplitude of the wave

    """
    # make plane wave
    E = np.exp(-1j * (k[0]*x + k[1]*y + k[2]*z))
    return E

def gaussian(x, y, z, w0, lamb):
    """Generate gaussian beam.

    :x: array of x values
    :y: array of y values
    :z: plane to evaluate field on
    :w0: beam waist
    :z0: Rayleigh range
    :lamb: wavelength
    :returns: complex amplitude of the wave

    """
    # wave number (propagating along z)
    k = 2*np.pi/lamb
    z0 = np.pi*w0**2/lamb
    w = lambda z: w0 * np.sqrt(1 + (z/z0)**2)
    R = lambda z: z*(1 + (z0/z)**2) if z != 0 else np.infty
    psi = lambda z: np.arctan(z/z0)
    E = w0 / w(z) \
        * np.exp(-(x**2 + y**2) / w(z)**2) \
        * np.exp(-1j*(k*z + k*(x**2+y**2)/(2*R(z)) - psi(z))) \
        * np.exp(1j*k*z)
    return E


def bessel(x, y, z, a, b):
    """Generate gaussian beam.

    :x: array of x values
    :y: array of y values
    :z: plane to evaluate field on
    :a: width parameter
    :b: kz component
    :returns: complex amplitude of the wave

    """
    # wave number (propagating along z)
    E = np.exp(1j * b*z) * j0(a * np.sqrt(x**2 + y**2))
    return E

