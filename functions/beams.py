import numpy as np
from scipy.special import j0, hankel1, hankel2, gamma, factorial
from scipy.special import eval_hermite as herm

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

def hankel(x, y, z, kr, kz, m=0, kind=1):
    """Generate Hankel waves.

    :x: array of x values
    :y: array of y values
    :z: plane to evaluate field on
    :kr: radial component of wave vector
    :kz: z component of wave vector
    :m: order of hankel functions
    :kind: 1 produces outgoing waves, 2 produces incoming waves, and 3 produces both
    :returns: complex amplitude of the wave
    """
    r = np.sqrt(x**2 + y**2)
    r[r == 0] = 1e-20 # to avoid divergence at origin
    phi = np.arctan2(y, x)

    E = np.exp(1j*(kz*z + m*phi))
    if kind==1:
        E = E*hankel1(m, kr*r)
    elif kind==2:
        E = E*hankel2(m, kr*r)
    elif kind==3:
        E = E * (hankel1(m, kr*r) + hankel2(m, kr*r)) / 2

    return E

def hermite_gauss(x, y, z, w0, lamb, m=0, n=0, norm=True):
    """Generate Hermite-Gaussian beams.

    :x: array of x values
    :y: array of y values
    :z: plane to evaluate field on
    :w0: beam waist
    :lamb: wavelength
    :m: order of x-dependent hermite function
    :n: order of y-dependent hermite function
    :norm: True for normalized HG functions, False for unnormalized
    :returns: complex amplitude of the wave
    """
    # rayleigh range
    z0 = np.pi*w0**2/lamb
    # gaussian field
    u = gaussian(x, y, z, w0, lamb)
    # waist
    w = w0 * np.sqrt(1 + (z/z0)**2)
    # phase
    phi = (m+n) * np.arctan(z/z0)
    # normalization
    scale = np.sqrt(2/np.pi) * 2**(-(m+n)/2) \
        / np.sqrt(factorial(n) * factorial(m) * w**2) \
        if norm else 1
    # field
    E = scale * herm(m, np.sqrt(2)*x/w) * herm(n, np.sqrt(2)*y/w) * u * np.exp(1j * phi)
    return E

