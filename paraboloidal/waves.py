import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.misc import derivative
from ..functions.paraboloidal import cart2pb, pb2cart
from ..functions.hypergeometric import P, X, L, pinney_wave

m = 5
n = 15

# compare pinney's function to X

r_start = 1
r_end = 20
r = np.linspace(r_start, r_end, num=200).reshape([1,-1])

X_eval = X(m, n, r, no_factorials=True)
P_eval = P(m, n, r, method='known_functions').astype(complex)
L_eval = L(n, m, r).flatten()

r, P_eval, X_eval = [xi.flatten() for xi in (r, P_eval, X_eval)]

plot1 = False
plot2 = False
plot3 = True

if plot1:

    fig, ax = plt.subplots(1,1)
    ax.plot(r, -L_eval, label='Laguerre de 1er tipo')
    ax.scatter(r, np.real(P_eval), label=r'Re{Pinney}')
    ax.plot(r, X_eval, label='Laguerre de 2o tipo')
    ax.scatter(r, np.imag(P_eval), label=r'Im{Pinney}')
    plt.legend()
    plt.show()

# see asymptotic behavior

P_large_n = P(m, n, r).astype(complex)
angle = (2*np.sqrt(n*r) - 0.5*np.pi*m - 0.5*np.pi)
#angle = 2*np.sqrt(n*r)
P_asymptotic = \
    1/np.sqrt(np.pi) \
    * n**(0.5*m-0.5) \
    * r**(-0.5*m-0.5) \
    * np.exp(0.5*r) \
    * (np.cos(angle) + 1j * np.sin(angle))

if plot2:

    fig, ax = plt.subplots(1,1)
    ax.plot(r, np.real(P_asymptotic), label='Comportamiento asintótico (real)')
    ax.scatter(r, np.real(P_large_n), label=f'Pinney con $n={n}$ (real)')
    ax.plot(r, np.imag(P_asymptotic), label='Comportamiento asintótico (imaginario)')
    ax.scatter(r, np.imag(P_large_n), label=f'Pinney con $n={n}$ (imaginario)')
    plt.legend()
    plt.show()

# calculate errors with differential equation

X_eval = X(m, n, r).astype(complex).flatten()

def laguerre_eq(m, n, z, kind='pinney'):
    # lambda to evaluate function
    if kind == 'laguerre':
        evaluate_U = lambda z: L(n, m, z)
    elif kind == 'X':
        evaluate_U = lambda z: X(m, n, z)
    elif kind == 'pinney':
        evaluate_U = lambda z: P(m, n, z)
    else:
        raise ValueError('invalid kind for diff. eq.')
    # differential equation
    dx = 1e-4
    return z * derivative(evaluate_U, z, n=2, dx=dx) \
        + (m + 1 - z) * derivative(evaluate_U, z, n=1, dx=dx) \
        + n * evaluate_U(z)

pin_error = laguerre_eq(m, n, r)
la1_error = laguerre_eq(m, n, r, kind='laguerre')
la2_error = laguerre_eq(m, n, r, kind='X')

if plot3:
    fig, ax = plt.subplots(1,1)
    ax.plot(r.flatten(),
            np.abs(pin_error.astype(complex).flatten()),
            label=r'$P_n^m$')
    ax.scatter(r.flatten(),
               np.abs(la1_error.astype(complex).flatten()),
               label=r'$L_n^m$')
    ax.scatter(r.flatten(),
               np.abs(la2_error.astype(complex).flatten()),
               label=r'$X_n^m$')
    #ax.plot(r, X_eval,
    #        label='Laguerre de 2o tipo')
    #ax.plot(r, np.abs(P_eval),
    #        label=r'abs{Pinney}')
    #ax.plot(r, L_eval,
    #        label='Laguerre de 1er tipo')
    #ax.plot(r, np.abs(L_eval + 1j*X_eval),
    #        label=r'$|L + iX|$')
    ax.set_xlabel('x')
    #ax.set_ylabel('ec. diferencial de Laguerre')
    fig.legend()
    plt.show()

