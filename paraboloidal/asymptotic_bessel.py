import numpy as np
import matplotlib.pyplot as plt

from ..functions.beams import pinney_wave
from ..functions.hypergeometric import besselj, H1, H2

n = 4
m = 2

x = np.linspace(0, 100, num=200).reshape([1,-1])

# parameter for asymptotic behavior

s = np.linspace(0.2, 0.5, num=15)

# expected asymptotic behavior

bessel_asymp = besselj(m, 2*np.sqrt(n*x)).flatten()
hankel_asymp = H2(m, 2*np.sqrt(n*x)).flatten()

# function for each value of n

S_asymp = [(si/n)**(0.5*m) * pinney_wave(n/si, m, si*x, kind='S') \
           .astype(complex).flatten() for si in s]
V_asymp = [(si/n)**(0.5*m) * pinney_wave(n/si, m, si*x, kind='V') \
           .astype(complex).flatten() for si in s]
x = x.flatten()

# colors for each plot

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    '''
    return plt.cm.get_cmap(name, n+1)

colors = get_cmap(s.size)

# plot

fig, ax = plt.subplots(1,1)
fig.suptitle(r'$S_{n/s}^m$ para $s \to 0$')

for i in range(s.size):
    label = f's={s[i]}' if (i == 0 or i == s.size-1) else '_'
    ax.plot(x, S_asymp[i], color=colors(i), alpha=0.8, label=label)

ax.plot(x, bessel_asymp, c='black', lw=2, linestyle='dashed', label='comportamiento asintótico')
fig.legend()

plt.show()

fig, ax = plt.subplots(1,1)
fig.suptitle(r'$V_{n/s}^m$ para $s \to 0$')

for i in range(s.size):
    label = f's={s[i]}' if (i == 0 or i == s.size-1) else '_'
    ax.plot(x, np.real(V_asymp[i]), color=colors(i), alpha=0.8, label=label)

ax.plot(x, np.real(hankel_asymp), c='black', lw=2, linestyle='dashed', label='comportamiento asintótico')
ax.set_ylim(-2, 2)
fig.legend()

plt.show()
