import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from ..functions.beams import pinney_wave

n = 10
m = 2

x = np.linspace(1, 10, num=200).reshape([1,-1])

# expected asymptotic behavior

S_expected = 1/(2*np.sqrt(np.pi)) * n**(m/2 - 1/4) \
    * np.exp(-1j * np.pi/8) * x**(-1/4) \
    * ( np.exp(-np.sqrt(2*n*x)) \
        * np.exp(1j*(np.sqrt(2*n*x) - np.pi*m/2 - np.pi/4)) \
       + np.exp(np.sqrt(2*n*x)) \
        * np.exp(-1j*(np.sqrt(2*n*x) - np.pi*m/2 - np.pi/4)) )
V_expected = 1/np.sqrt(np.pi) * n**(m/2 - 1/4) \
    * np.exp(1j*np.pi/8) * x**(-1/4) \
    * np.exp(1j*(np.sqrt(2*n*x) - m*np.pi/2 - np.pi/4))
    #* np.exp(-np.sqrt(2*n*x)) \

S_expected = S_expected.flatten()
V_expected = V_expected.flatten()

# function

S_asymp = pinney_wave(n, m, 1j*x, kind='S').astype(complex).flatten()
V_asymp = pinney_wave(n, m, 1j*x, kind='V').astype(complex).flatten()
x = x.flatten()

# plot for S

fig, ax = plt.subplots(1,1)
fig.set_size_inches(w=2.5, h=2)
ax.plot(x, x**(-3)/1000 * np.real(S_expected), c='black', linestyle='dashed')
ax.plot(x, x**(-3)/1000 * np.imag(S_expected), c='red', linestyle='dashed')
ax.plot(x, x**(-3)/1000 * np.real(S_asymp), c='black')
ax.plot(x, x**(-3)/1000 * np.imag(S_asymp), c='red')
ax.set_xlabel('$x$')
ax.set_ylabel(r'$S_\nu^\mu / x^3 1000$')
#plt.savefig('S_large_nu.pdf', bbox_inches='tight')
plt.show()


# plot for V

fig, ax = plt.subplots(1,1)
fig.set_size_inches(w=2.5, h=2)
ax.plot(x, np.real(V_expected), c='black', linestyle='dashed')
ax.plot(x, np.imag(V_expected), c='red', linestyle='dashed')
ax.plot(x, np.exp(-np.sqrt(2*n*x)) * np.real(V_asymp), c='black')
ax.plot(x, np.exp(-np.sqrt(2*n*x)) * np.imag(V_asymp), c='red')
ax.set_xlabel('$x$')
ax.set_ylabel(r'$V_\nu^\mu / x^3 1000$')
ax.set_ylim(-10, 10)
plt.savefig('V_large_nu.pdf', bbox_inches='tight')
plt.show()

