import numpy as np

def mc_integrate(x0, xf, f, n=1e4):
    """Integrate 1D function using Monte Carlo.

    :x0: lower bound for integral
    :xf: upper bound for integral
    :f: function to perform integration on
    :n: amount of samples to use on integration
    :returns: integral of f from x0 to xf

    """

    # generate random numbers from even distribution
    u = np.random.default_rng().uniform(x0, xf, size=(int(n), 1))
    # evaluate function
    fu = f(u)

    # estimate integral
    I = (xf - x0) * np.mean(fu)

    return I

