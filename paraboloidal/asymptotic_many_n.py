import numpy as np
import matplotlib.pyplot as plt

from ..functions.special import P

m = 5
n = np.arange(35, 36, 5)

r_start = 1
r_end = 10

r = np.linspace(r_start, r_end, num=200).reshape([1,-1])
P_eval = [laguerreP(m, ni, r, method='sums').astype(complex) for ni in n]

r = r.flatten()
P_eval = [P_eval_i.flatten() for P_eval_i in P_eval]

# asymptotic behavior

ch_sign = np.ones(r.shape)
ch_sign[np.sign(r) != -1] = 1

n = n.reshape([-1,1])
asymp_coef = (1/np.sqrt(np.pi) \
              * n**(0.5*m - 0.25) * r**(-0.5*m - 0.25) \
              * np.exp(0.5*r))
ang = 2*np.sqrt(n*r) - 0.5*np.pi*m - (3/4)*np.pi
asymp_re = asymp_coef * np.cos(ang)
asymp_im = asymp_coef * np.sin(ang)
n = n.flatten()

comparison_P = P_eval

# plotting

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    '''
    return plt.cm.get_cmap(name, n+1)

n_plot = 5
colors = get_cmap(n_plot)
idx = np.arange(0, r.size, 5)

fig_count = 0
fig, ax = plt.subplots(1,2)
fig.set_size_inches(12, 5)
ax[0].set_ylabel('Re{$P_n^m$}')
ax[1].set_ylabel('Im{$P_n^m$}')
for i in range(n.size):

    color = colors(i%n_plot)

    # real plot
    ax[0].scatter(r[idx], np.real(comparison_P[i][idx]), s=1, color=color)
    ax[0].plot(r, asymp_re[i], color=color, lw=1)

    # imaginary plot
    ax[1].scatter(r[idx], np.imag(comparison_P[i][idx]), label=f'$n={n[i]}$', s=1, color=color)
    ax[1].plot(r, asymp_im[i], color=color, lw=1)

    # new plot every n_plot values of n
    if (i % n_plot == n_plot-1) or (i == n.size-1):
        fig_count += 1
        fig.legend()
        plt.savefig(f'asymp{fig_count}.pdf', bbox_inches='tight')
        plt.show()
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(12, 5)
        ax[0].set_ylabel('Re{$P_n^m$}')
        ax[1].set_ylabel('Im{$P_n^m$}')
