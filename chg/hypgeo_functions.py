import mpmath as mp
import numpy as np
from scipy.special import factorial, gamma
from scipy.misc import derivative
from math_functions import converge_sum, integer_stream

def mp_wrapper(mp_func, x, *args, **kwargs):
    """Wrap an mp function for numpy arrays.

    :mp_func: function to wrap
    :shape: argument's shape
    :args: arguments and parameters for mp function
    :returns: mp_func(*args, **kwargs) as an ndarray

    """
    sh = x.shape
    y = [mp_func(*args, xi) for xi in x.flatten()]
    y = np.array(y).reshape(sh)
    return y

def digamma(x):
    """Wrap mpmath's digamma for numpy arrays.

    :x: ndarray argument
    :returns: Psi(x)
    """
    return mp_wrapper(mp.digamma, x)

def L(n, a, z):
    """Wrap mpmath's laguerre for numpy arrays.

    :n: first parameter for laguerre function
    :a: second parameter for laguerre function
    :z: argument for laguerre function
    :returns: L_n^a(z)

    """
    return mp_wrapper(mp.laguerre, z, n, a)

def M(a, b, z):
    """Wrap mpmath's 1F1 for numpy arrays.

    :a: first parameter
    :b: second parameter
    :z: ndarray argument
    :returns: M(a, b, z) as a ndarray

    """
    sh = z.shape
    y = [mp.hyp1f1(a, b, zi) for zi in z.flatten()]
    y = np.array(y).reshape(sh)
    return y

def U(a, b, z):
    """Wrap mpmath's hyperu for numpy arrays.

    :a: first parameter
    :b: second parameter
    :z: ndarray argument
    :returns: U(a, b, z) as a ndarray

    """
    sh = z.shape
    y = [mp.hyperu(a, b, zi) for zi in z.flatten()]
    y = np.array(y).reshape(sh)
    return y

def X(m, n, x, method='sums'):
    """Compute Laguerre's function of the second kind.

    :m: first parameter
    :n: second parameter
    :x: real argument
    :returns: X_n^m(x)

    """
    # constant
    N = n + (m+1)/2
    # calculate coefficient
    Cmn = (-1)**(n+1) * factorial(n+m) * N**(-m) \
        / (factorial(m) * factorial(n)**2)

    # functions for each term
    term1_func = lambda k: factorial(k-1) \
        / (factorial(n+k)*factorial(m-k)) * np.power(x, -k)
    term2_func = lambda k: factorial(k-n-1) \
        / (factorial(m+k)*factorial(k)) * np.power(x, k)
    term3_func = lambda k: np.power(-x, k) \
        / (factorial(n-k)*factorial(m+k)*factorial(k)) \
        * (-np.log(x) + digamma(m+k+1) \
           + digamma(k+1) - digamma(n-k+1))

    if method == 'derivatives':
        # calculate first term
        term1_func = lambda nu: gamma(nu+m+1) * M(-nu, m+1, x)
        term1 = derivative(term1_func, n)
        # calculate second term
        term2_func = lambda nu: U(-nu, m+1, x)
        term2 = -factorial(m)/np.pi * derivative(term2_func, n)
        # sum
        y = Cmn * (-1)**n/np.pi * (term1 + term2)

    elif method == 'og':
        y = Cmn * (gamma(n+m+1)*M(-n,m+1,x) \
                   - factorial(m) * np.cos(n*np.pi) * U(-n,m+1,x)) \
            / np.sin(n*np.pi)

    elif method == 'sums':

        # evaluate first and last term

        term1_k = np.arange(1, m+1).reshape([-1, 1])
        term3_k = np.arange(0, n+1).reshape([-1, 1])

        term1 = term1_func(term1_k).sum(axis=0, keepdims=True)
        term3 = term3_func(term3_k).sum(axis=0, keepdims=True)

        # evaluate second term
        term2_k = integer_stream(n+1, np.infty, shape=[-1, 1])
        term2 = (-1)**(n+1) * converge_sum(term2_func, term2_k, keepdims=True)

        # add them
        y = Cmn * (-1)**n * factorial(n) * factorial(m) \
            * factorial(n+m) / np.pi * (term1 + term2 + term3)

    elif 'bad sums':

        # indices for each term
        term1_k = np.arange(1, m+1).reshape([-1, 1])
        term2_k = np.arange(n+1, n+50).reshape([-1, 1])
        term3_k = np.arange(0, n+1).reshape([-1, 1])

        # evaluate each term
        term1 = term1_func(term1_k).sum(axis=0, keepdims=True)
        term2 = (-1)**(n+1) * term2_func(term2_k).sum(axis=0, keepdims=True)
        term3 = term3_func(term3_k).sum(axis=0, keepdims=True)

        # add them
        y = Cmn * (-1)**n * factorial(n) * factorial(m) \
            * factorial(n+m) / np.pi * (term1 + term2 + term3)

    else:
        raise ValueError('invalid method name')

    return y, Cmn

