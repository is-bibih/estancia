import mpmath as mp
import numpy as np
from scipy.special import factorial, gamma, binom, hankel1, hankel2
from scipy.misc import derivative
from scipy.integrate import quad_vec
from .summation import converge_sum, integer_stream
from .integration import mc_integrate

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
    y = y.astype(complex)
    if not np.any(np.iscomplex(y)):
        y = y.astype(float)
    return y

def digamma(x):
    """Wrap mpmath's digamma for numpy arrays.

    :x: ndarray argument
    :returns: Psi(x)
    """
    return mp_wrapper(mp.digamma, x)

def besselj(n, x):
    """Wrap mpmath's besselj for numpy arrays.

    :n: order
    :x: ndarray argument
    :returns: J_n(x)

    """
    return mp_wrapper(mp.besselj, x, n)

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

def H1(m, z):
    """Wrap mpmath's hankel1 for numpy arrays.

    :m: order for the Hankel function
    :z: ndarray argument
    :returns:

    """
    return mp_wrapper(mp.hankel1, z, m).astype(complex)

def H2(m, z):
    """Wrap mpmath's hankel2 for numpy arrays.

    :m: order for the Hankel function
    :z: ndarray argument
    :returns:

    """
    return mp_wrapper(mp.hankel2, z, m).astype(complex)

def X(m, n, x, method='sums', return_C=False, no_factorials=False):
    """Compute Laguerre's function of the second kind.

    :m: first parameter
    :n: second parameter
    :x: real argument
    :return_C_: whether the coefficients should be returned
    :no_factorials: whether leading factorial coefficient should be calculated; True for use in calculating Pinney's function
    :returns: X_n^m(x) or X_n^m(x), Cmn if return_C

    """

    sh = x.shape

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

        x_dims = x.ndim
        k_dims = [-1] + [1 for i in range(x_dims)]
        x = np.expand_dims(x, axis=0)

        term1_k = np.arange(1, m+1).reshape(k_dims)
        term3_k = np.arange(0, n+1).reshape(k_dims)

        term1 = term1_func(term1_k).sum(axis=0, keepdims=True)
        term3 = term3_func(term3_k).sum(axis=0, keepdims=True)

        # evaluate second term
        term2_k = integer_stream(n+1, np.infty, shape=k_dims)
        term2 = (-1)**(n+1) * converge_sum(term2_func, term2_k, keepdims=True)

        # add them
        if no_factorials:
            y = factorial(n+m) / np.pi * (term1 + term2 + term3)
        else:
            y = Cmn * (-1)**n * factorial(n) * factorial(m) \
                * factorial(n+m) / np.pi * (term1 + term2 + term3)

        y = np.squeeze(y)

    elif method == 'bad sums':

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
        raise ValueError('invalid method name for calculation of X')

    if return_C:
        return y.reshape(sh), Cmn
    else:
        return y.reshape(sh)

def P(m, n, z, method='known_functions', n_integrate=1e4):
    """Evaluate Pinney's function.

    :m: first parameter for laguerre function
    :n: second parameter for laguerre function
    :z: argument for Pinney's function
    :method: can be 'known_functions', 'sums', or 'integrals'
    :n_integrate: amount of points to use in case of numeric integration
    :returns: U_n^m(z)

    """

    sh = z.shape

    if method == 'known_functions':

        y = -L(n, m, z) + 1j * X(m, n, z, no_factorials=True)

    elif method == 'sums' or method == 'mpsums':

        # sign according to z's angle
        change_sign = np.ones(z.shape)
        change_sign[np.imag(z) < 0] = -1

        # useful for summations
        harmonic = lambda p: digamma(p+1) - np.euler_gamma

        # functions for each sum

        sum1_func = lambda p: factorial(m+n) * factorial(p-1) \
            / (factorial(p+n) * factorial(m-p)) * z**(-p)
        sum2_func = lambda p: binom(n, p) \
            * (np.log(z) + np.euler_gamma - change_sign * np.pi * 1j \
               + harmonic(n-p) - harmonic(p) - harmonic(p+m)) \
            * (-z)**p / factorial(p+m)
        sum3_func = lambda p: factorial(p-n-1) * z**p \
            / (factorial(p+m) * factorial(p))

        if method == 'sums':
            # indices for each sum

            sum1_p = np.arange(1, m+1).reshape([-1, 1])
            sum2_p = np.arange(0, n+1).reshape([-1, 1])
            sum3_p = integer_stream(n+1, np.infty, shape=[-1, 1])

            # evaluate each sum

            sum1 = sum1_func(sum1_p).sum(axis=0, keepdims=True)
            sum2 = sum2_func(sum2_p).sum(axis=0, keepdims=True)
            sum3 = converge_sum(sum3_func, sum3_p, keepdims=True)

        else:

            sum1 = mp.nsum(sum1_func, [1, m])
            sum2 = mp.nsum(sum2_func, [0, n])
            sum3 = mp.nsum(sum3_func, [n+1, mp.inf])

        # put everything together

        y = - 1j / np.pi * change_sign * (- sum1 \
             + factorial(m+n)/factorial(n) * sum2 \
             + (-1)**n * factorial(n+m) * sum3)

    elif method == 'integrals':

        # reshape z so integrals don't try to broadcast
        sh = z.shape
        z = np.reshape(z, (1, -1))

        # find where to use each integral
        pos_imag = np.imag(z) >= 0
        not_pos_imag = np.logical_not(pos_imag)

        # define functions for integral

        # for Im(z) > 0
        #integral_func_1 = lambda t: np.exp(-t) \
        #    * t ** (0.5*m + n) \
        #    * H1(m, 2 * np.sqrt(z[pos_imag]*t))
        #integral_func_1_tau = lambda tau: np.exp(-1/tau) \
        #    * tau**(-m/2 - n - 2) \
        #    * H1(m, 2*np.sqrt(z[pos_imag]/tau))
        integral_func_1 = lambda t: np.exp(-t) \
            * t ** (0.5*m + n) \
            * H1(m, 2 * np.sqrt(z[pos_imag]*t)) \
            + np.exp(-1/t) \
            * t**(-0.5*m - n - 2) \
            * H1(m, 2*np.sqrt(z[pos_imag]/t))

        # for Im(z)  < 0
        #integral_func_2 = lambda t: np.exp(-t) \
        #    * t ** (0.5*m + n) \
        #    * H2(m, 2 * np.sqrt(z[not_pos_imag]*t))
        #integral_func_2_tau = lambda tau: np.exp(-1/tau) \
        #    * tau**(-m/2 - n - 2) \
        #    * H2(m, 2*np.sqrt(z[not_pos_imag]/tau))
        integral_func_2 = lambda t: np.exp(-t) \
            * t ** (0.5*m + n) \
            * H2(m, 2 * np.sqrt(z[not_pos_imag]*t)) \
            + np.exp(-1/t) \
            * t**(-0.5*m - n - 2) \
            * H2(m, 2*np.sqrt(z[not_pos_imag]/t))

        # leading coefficient for y
        y = (z**(m/2) * np.exp(z) / gamma(n+1)).astype(complex)

        # evaluate each integral
        #y[pos_imag] *= mc_integrate(0, 1, integral_func_1, n=n_integrate)
        #y[not_pos_imag] *= mc_integrate(0, 1, integral_func_2, n=n_integrate)
        y[pos_imag] *= quad_vec(integral_func_1, 1e-4, 1)[0]
        y[not_pos_imag] *= quad_vec(integral_func_2, 1e-4, 1)[0]

    elif method == 'recursive':

        # sign change
        ch_sign = np.sign(z)
        ch_sign[ch_sign != -1] = 1

        # function to get P recursively
        y_func = lambda p: 1/ L(n-1, m, z) \
                * ( 1j / np.pi * gamma(m+n)/gamma(n) \
                    * z**(-m) * np.exp(z) \
                   + p * L(n, m, z))
        if (n > 2) and np.all(np.real(z) > -1):
            p = P(m, n-1, z, method='recursive')
            y = y_func(p).astype(complex)
            y[np.isnan(y)] = 0
        elif (n > 2) and not np.all(np.real(z) <= -1):
            raise ValueError('argument must have real part greater than -1')
        else:
            p = P(m, n-1, z, method='sums')
            y = y_func(p)

    else:
        raise ValueError('invalid method for calculation of Pinney function')

    return y.reshape(sh)

def pinney_wave(n, m, z, kind='S'):
    """Evaluate Pinney's S and V functions

    :n: parameter n of function
    :m: parameter m of function
    :z: argument of function
    :kind: may be either S, V or W
    :returns: either S_n^m(z), V_n^m(z), or W_n^m(z), according to kind

    """
    if kind == 'S':
        return z**(0.5*m) * np.exp(-0.5*z) * L(n, m, z)
    elif kind == 'V':
        return z**(0.5*m) * np.exp(-0.5*z) * P(n, m, z)
    elif kind == 'W':
        return z**(0.5*m) * np.exp(-0.5*z) * X(n, m, z)
    else:
        raise ValueError('invalid kind for Pinney wave function')

