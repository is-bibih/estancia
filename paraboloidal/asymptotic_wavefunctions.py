import numpy as np
import matplotlib.pyplot as plt

from ..functions.special import parabV, parabS, parabW

k = 2*np.pi / 633e-9
k = 1

n = 25
m = 8

x_start = 5
x_end = 20

x = np.linspace(x_start, x_end, num=200).reshape([1,-1])
z = 2j*k*x
#z = x

# pinney wave
V = parabV(n, m, z, ).astype(complex).flatten()

# asymptotic behavior
sign = np.ones(z.shape)
sign[np.angle(z) < 0] = -1
V_asymp = (1/np.sqrt(np.pi) * n**(0.5*m - 0.25) * z**(-0.25) \
    * np.exp(1j * sign * (np.sqrt(n*z) - 0.5*np.pi*m - 0.25*np.pi))).flatten()
#V_asymp = (1/np.sqrt(np.pi) * n**(m/2 - 0.25) * np.exp(1j*np.pi/8) \
#           * x**(-0.25) * np.exp(-np.sqrt(2*n*x)) \
#           * np.exp(1j*(np.sqrt(2*n*x) - 0.5*np.pi*m - 0.25*np.pi))).flatten()

x = x.flatten()

fig, ax = plt.subplots(1, 2)
ax[0].plot(x, np.real(V), label='V')
ax[1].plot(x, np.imag(V), label='V')
ax[0].plot(x, np.real(V_asymp), label='asymptotic')
ax[1].plot(x, np.imag(V_asymp), label='asymptotic')
ax[0].set_ylim(-5, 5)
ax[1].set_ylim(-5, 5)
plt.legend()
plt.show()

