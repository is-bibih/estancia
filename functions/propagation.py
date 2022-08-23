import numpy as np
from numpy.fft import fft2, ifft2, fftshift, fftfreq

def beam_propagation_method(E0, x, y, zf, lamb):
    """Propagate a field given an initial complex amplitude.

    :E0: Initial complex amplitude
    :x: Array of x coordinates
    :y: Array of y coordinates
    :zf: Distance to propagate to
    :lamb: Wavelength
    :returns: Array with final complex amplitude

    """
    dx = x.flatten()[1] - x.flatten()[0]
    dy = y.flatten()[1] - y.flatten()[0]
    delta_x = x.flatten()[-1] - x.flatten()[0]
    delta_y = y.flatten()[-1] - y.flatten()[0]
    # spatial frequencies
    fx = fftfreq(x.size, d=dx).reshape(x.shape)
    fy = fftfreq(y.size, d=dy).reshape(y.shape)
    dfx = fx.flatten()[1] - fx.flatten()[0]
    dfy = fy.flatten()[1] - fy.flatten()[0]
    # directing cosines
    a = fx * lamb
    b = fy * lamb
    # check not evanescent region (frequency space)
    valid_region = a**2 + b**2 <= 1
    # angular spectrum at origin
    A0 = fft2(E0, axes=(0, 1))
    # angular spectrum at final plane
    Af = A0 * np.exp(1j * 2*np.pi/lamb \
                     * np.sqrt(1 - a**2 - b**2) * zf)
    # complex frequency at final plane
    Ef = ifft2(Af * valid_region, axes=(0, 1))
    return Ef

